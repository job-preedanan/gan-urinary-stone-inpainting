from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import tensorflow as tf
from cascaded_dilatedNet_gan import DilatedNetGANs

import matplotlib.pyplot as plt
import numpy as np
import cv2

IMAGE_ROW = 128
IMAGE_COL = 128


def build_generator(img_shape=(IMAGE_ROW, IMAGE_COL, 1), output_ch=1):
    def conv2d(layer_input, filters, f_size=3, strides=1, d_rate=1, bn=False):
        d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', dilation_rate=d_rate)(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)

        return d

    def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0, bn=False):
        u = Conv2DTranspose(filters, kernel_size=2, strides=2, kernel_initializer='he_uniform')(layer_input)
        u = Activation(activation='relu')(u)
        u = BatchNormalization(momentum=0.8)(u)

        u = Concatenate()([u, skip_input])
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        if bn:
            u = BatchNormalization(momentum=0.8)(u)

        return u

    def dilated_net(d0, gf=64):

        # 128 x 128
        d1_1 = conv2d(d0, gf, f_size=3, strides=1, d_rate=1)
        d1_2 = conv2d(d1_1, gf, f_size=3, strides=1, d_rate=1)

        # 64 x 64
        d2_1 = conv2d(d1_2, gf * 2, f_size=3, strides=2, d_rate=1)
        d2_2 = conv2d(d2_1, gf * 2, f_size=3, strides=1, d_rate=1)

        # 32 x 32
        d3_1 = conv2d(d2_2, gf * 4, f_size=3, strides=2, d_rate=1)
        d3_2 = conv2d(d3_1, gf * 4, f_size=3, strides=1, d_rate=1)
        d3_3 = conv2d(d3_2, gf * 4, f_size=3, strides=1, d_rate=1)

        d3_4 = conv2d(d3_3, gf * 4, f_size=3, strides=1, d_rate=2)
        d3_5 = conv2d(d3_4, gf * 4, f_size=3, strides=1, d_rate=4)
        d3_6 = conv2d(d3_5, gf * 4, f_size=3, strides=1, d_rate=8)

        d3_8 = conv2d(d3_6, gf * 4, f_size=3, strides=1, d_rate=1)
        d3_9 = conv2d(d3_8, gf * 4, f_size=3, strides=1, d_rate=1)

        # 64 x 64
        u2 = deconv2d(d3_9, d2_2, gf * 2)

        # 128 x 128
        u1 = deconv2d(u2, d1_2, gf)

        return u1

    # Image input
    d0_1 = Input(shape=img_shape)

    d0_2 = dilated_net(d0_1, 64)
    u0_1 = Conv2D(output_ch, kernel_size=3, strides=1, padding='same', activation='tanh')(d0_2)

    u1_2 = dilated_net(u0_1, 64)
    u0_2 = Conv2D(output_ch, kernel_size=3, strides=1, padding='same', activation='tanh')(u1_2)

    return Model(d0_1, [u0_1, u0_2])


def load_images(images_list, img_rows=IMAGE_ROW, img_cols=IMAGE_COL):

    images = np.zeros((len(images_list), img_rows, img_cols, 1), np.float32)
    masks = np.zeros((len(images_list), img_rows, img_cols, 1), np.float32)

    for i, image_name in enumerate(images_list):

        # load image
        image = cv2.imread('images/' + image_name, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (img_rows, img_cols))
        image = image[:, :, np.newaxis]
        images[i] = image / 127.5 - 1

        # load masks
        mask = cv2.imread('masks/' + image_name, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMAGE_ROW, IMAGE_COL))
        mask = mask[:, :, np.newaxis]
        masks[i] = mask / 127.5 - 1

    return images, masks


def predict(samples):
    from matplotlib.colors import NoNorm
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

    # Load the dataset
    x_test, y_test = load_images(samples)

    inpainting_generator = build_generator()
    inpainting_generator.load_weights('saved_weights/generator_weights.hdf5')
    [gen_images1, gen_images] = inpainting_generator.predict(y_test, batch_size=16)

    ssim_values = np.zeros(len(gen_images))
    mse_values = np.zeros(len(gen_images))
    psnr_values = np.zeros(len(gen_images))
    for i in range(len(gen_images)):
        org_image = 0.5 * np.squeeze(x_test[i]) + 0.5
        mask_image = 0.5 * np.squeeze(y_test[i]) + 0.5
        gen_image = 0.5 * np.squeeze(gen_images[i]) + 0.5

        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(org_image, cmap="gray", norm=NoNorm())
        axs[0].axis('off')
        axs[1].imshow(mask_image, cmap="gray", norm=NoNorm())
        axs[1].axis('off')
        axs[2].imshow(gen_image, cmap="gray", norm=NoNorm())
        axs[2].axis('off')

        ssim_values[i] = ssim(org_image, gen_image)
        mse_values[i] = mean_squared_error(org_image, gen_image)
        psnr_values[i] = peak_signal_noise_ratio(org_image, gen_image)

        plt.show()

        # fig.savefig(cfg.save_path + os.sep + cfg.experiment_name +
        #             '/test_results/' + samples[i][0][:-4] + '_result.png')
        # plt.close()

    mean_ssim = np.mean(ssim_values)
    mean_mse = np.mean(mse_values)
    mean_psnr = np.mean(psnr_values)

    return mean_mse, mean_psnr, mean_ssim


if __name__ == '__main__':
    import glob

    test_samples = glob.glob1('images/', '*')
    predict(samples=test_samples)

