from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import tensorflow as tf
from cascased_dilatedNet_gan import DilatedNet

import matplotlib.pyplot as plt
import numpy as np
import config as cfg
from pandas import DataFrame
import cv2

IMAGE_ROW = 128
IMAGE_COL = 128


def load_images(images_list, img_rows=IMAGE_ROW, img_cols=IMAGE_COL, vgg_preprocessing=False):

    if vgg_preprocessing:
        images = np.zeros((len(images_list), 256, 256, 3), np.float32)
    else:
        images = np.zeros((len(images_list), img_rows, img_cols, 1), np.float32)

    masks = np.zeros((len(images_list), img_rows, img_cols, 1), np.float32)

    for i, image_name in enumerate(images_list):

        # IMAGE_DIR
        if image_name[1] == '1':  # stone-contained
            mask_dir = SC_MASK_DIR
            image_dir = SC_ORG_DIR
        elif image_name[1] == '0':  # stone-free
            mask_dir = SF_MASK_DIR
            image_dir = SF_ORG_DIR

        # read image amd mask
        if vgg_preprocessing:
            image = cv2.imread(image_dir + os.sep + image_name[0])
            image = cv2.resize(image, (256, 256))
        else:
            image = cv2.imread('image' + image_name[0], cv2.IMREAD_GRAYSCALE)
            try:
        image = cv2.resize(image, (img_rows, img_cols))
        image = image[:, :, np.newaxis]
        images[i] = image / 127.5 - 1

        mask = cv2.imread('mask/' + image_name[0], cv2.IMREAD_GRAYSCALE)
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

    inpainting_generator = DilatedNet()

    generator.load_weights('/saved_model/generator_weights.hdf5')
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

        fig.savefig(cfg.save_path + os.sep + cfg.experiment_name +
                    '/test_results/' + samples[i][0][:-4] + '_result.png')
        plt.close()

    mean_ssim = np.mean(ssim_values)
    mean_mse = np.mean(mse_values)
    mean_psnr = np.mean(psnr_values)

    return mean_mse, mean_psnr, mean_ssim


if __name__ == '__main__':

    predict(samples=test_samples)

