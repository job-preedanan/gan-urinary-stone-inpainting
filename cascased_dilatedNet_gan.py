from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
import tensorflow as tf


class DilatedNetGANs(object):
    def __init__(self):
        self.img_rows = 128
        self.img_cols = 128
        self.crop_img_rows = 48
        self.crop_img_cols = 48
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.crop_img_shape = (self.crop_img_rows, self.crop_img_cols, self.channels)

        # Calculate output shape of global D (PatchGAN)
        global_patch = int(self.img_rows / 2**4)
        self.global_disc_patch = (global_patch, global_patch, 1)

        # Calculate output shape of local D (PatchGAN)
        local_patch = int(self.crop_img_rows / 2**3)
        self.local_disc_patch = (local_patch, local_patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the local discriminator
        self.local_discriminator = self.build_local_discriminator()
        self.local_discriminator.compile(loss=['mse'],
                                         optimizer=optimizer,
                                         metrics=['accuracy'])

        # Build and compile the global discriminator
        self.global_discriminator = self.build_global_discriminator()
        self.global_discriminator.compile(loss='mse',
                                          optimizer=optimizer,
                                          metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.summary()

        # Use a pre-trained VGG19 model to extract image features
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

        # Input images and their conditioning images
        images = Input(shape=self.img_shape)
        masks = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of org stone
        [gen_images1, gen_images] = self.generator(masks)

        # vgg loss
        # output for vgg loss
        gen_images_256 = UpSampling2D(size=2)(gen_images)    # 256 * 256
        gen_images_vgg = Concatenate()([gen_images_256, gen_images_256, gen_images_256])
        gen_features1, gen_features2, gen_features3, gen_features4 = self.vgg(gen_images_vgg)

        # For the combined model we will only train the generator
        self.local_discriminator.trainable = False
        self.global_discriminator.trainable = False

        # cropped center region for local discriminator (48 x 48)
        crop_gen_images = gen_images[:, 40:88, 40:88, :]
        crop_masks = masks[:, 40:88, 40:88, :]

        # Discriminators determines validity of translated images / condition pairs
        local_valid = self.local_discriminator([crop_gen_images, crop_masks])
        global_valid = self.global_discriminator([gen_images, masks])

        # weight map * img
        # w_gen_images = self.gm_weight(images) * gen_images
        mag_gens = self.image_gradient_magnitude(gen_images)

        self.combined = Model(inputs=[images, masks], outputs=[local_valid,
                                                               global_valid,
                                                               gen_images1,
                                                               mag_gens,
                                                               gen_features1,
                                                               gen_features2,
                                                               gen_features3,
                                                               gen_features4])

        self.combined.compile(loss=['mse', 'mse', 'mae', 'mse', 'mae', 'mae', 'mae', 'mae'],
                              loss_weights=[cfg.loss_weight['lD_weight'],
                                            cfg.loss_weight['gD_weight'],
                                            cfg.loss_weight['L1_weight'],
                                            cfg.loss_weight['L1_weight'],
                                            cfg.loss_weight['vgg_weight'],
                                            cfg.loss_weight['vgg_weight'],
                                            cfg.loss_weight['vgg_weight'],
                                            cfg.loss_weight['vgg_weight']],
                              optimizer=optimizer)

    def build_generator(self):

        def conv2d(layer_input, filters, f_size=3, strides=1, d_rate=1, bn=False):
            d = Conv2D(filters, kernel_size=f_size, strides=strides, padding='same', dilation_rate=d_rate)(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)

            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0, bn=False):
            u = Conv2DTranspose(filters, kernel_size=2, strides=2, kernel_initializer='he_uniform')(layer_input)
            u = Activation(activation='relu')(u)
            # u = BatchNormalization(momentum=0.8)(u)

            u = Concatenate()([u, skip_input])
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            if bn:
               u = BatchNormalization(momentum=0.8)(u)

            return u

        def dilated_net(d0):
            # 128 x 128
            d1_1 = conv2d(d0, self.gf, f_size=3, strides=1, d_rate=1)
            d1_2 = conv2d(d1_1, self.gf, f_size=3, strides=1, d_rate=1)

            # 64 x 64
            d2_1 = conv2d(d1_2, self.gf * 2, f_size=3, strides=2, d_rate=1)
            d2_2 = conv2d(d2_1, self.gf * 2, f_size=3, strides=1, d_rate=1)

            # 32 x 32
            d3_1 = conv2d(d2_2, self.gf * 4, f_size=3, strides=2, d_rate=1)
            d3_2 = conv2d(d3_1, self.gf * 4, f_size=3, strides=1, d_rate=1)
            d3_3 = conv2d(d3_2, self.gf * 4, f_size=3, strides=1, d_rate=1)

            d3_4 = conv2d(d3_3, self.gf * 4, f_size=3, strides=1, d_rate=2)
            d3_5 = conv2d(d3_4, self.gf * 4, f_size=3, strides=1, d_rate=4)
            d3_6 = conv2d(d3_5, self.gf * 4, f_size=3, strides=1, d_rate=8)

            d3_8 = conv2d(d3_6, self.gf * 4, f_size=3, strides=1, d_rate=1)
            d3_9 = conv2d(d3_8, self.gf * 4, f_size=3, strides=1, d_rate=1)

            # 64 x 64
            u2 = deconv2d(d3_9, d2_2, self.gf * 2)

            # 128 x 128
            u1 = deconv2d(u2, d1_2, self.gf)

            return u1

        # Image input
        d0_1 = Input(shape=self.img_shape)

        d0_2 = dilated_net(d0_1)
        u0_1 = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(d0_2)

        u1_2 = dilated_net(u0_1)
        u0_2 = Conv2D(self.channels, kernel_size=3, strides=1, padding='same', activation='tanh')(u1_2)

        return Model(d0_1, [u0_1, u0_2])

    def build_local_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.crop_img_shape)
        img_B = Input(shape=self.crop_img_shape)

        # Concatenate image and conditioning image by channels
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])  # 48

        d1 = d_layer(combined_imgs, self.df, bn=False)  # 24
        d2 = d_layer(d1, self.df * 2)  # 12
        d3 = d_layer(d2, self.df * 2)  # 6

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d3)

        return Model([img_A, img_B], validity)

    def build_global_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])  # 128

        d1 = d_layer(combined_imgs, self.df, bn=False)  # 64
        d2 = d_layer(d1, self.df * 2)  # 32
        d3 = d_layer(d2, self.df * 4)  # 16
        d4 = d_layer(d3, self.df * 8)  # 8

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)

    def image_gradient_magnitude(self, img):
        gY, gX = tf.image.image_gradients(tf.convert_to_tensor(img))
        return tf.sqrt(gY ** 2 + gX ** 2)

    def gradient_loss(self, a, b):
        mag_a = self.image_gradient_magnitude(a)
        mag_b = self.image_gradient_magnitude(b)

        return tf.reduce_mean((mag_a - mag_b) ** 2)

    def gmsim(self, a, b):
        gm_a = self.image_gradient_magnitude(a)
        gm_b = self.image_gradient_magnitude(b)
        sim_map = (gm_a * gm_b + 0.1)/(gm_a ** 2 + gm_b ** 2 + 0.1)

        return tf.reduce_mean(tf.boolean_mask(sim_map, tf.math.is_nan(sim_map)))

    def gmsim_loss(self, a , b):
        return 1.0 - self.gmsim(a, b)

    def gm_weight(self, img):
        w_map = self.image_gradient_magnitude(img)
        w_map = w_map / tf.reduce_sum(w_map)
        return w_map

    def gmw_mae_loss(self, ref, gen):
        w_map = self.image_gradient_magnitude(ref)
        w_map = w_map / tf.reduce_sum(w_map)
        return tf.reduce_mean(w_map * tf.abs(ref - gen))

    def build_vgg(self):

        vgg = VGG19(weights='imagenet', include_top=False)
        vgg.trainable = False

        img_feature1 = vgg.get_layer('block1_conv1').output
        img_feature2 = vgg.get_layer('block2_conv1').output
        img_feature3 = vgg.get_layer('block3_conv1').output
        img_feature4 = vgg.get_layer('block4_conv1').output

        return Model(inputs=vgg.input, outputs=[img_feature1, img_feature2, img_feature3, img_feature4])

    def load_images(self, images_list, vgg_preprocessing=False):
        import cv2

        if vgg_preprocessing:
            images = np.zeros((len(images_list), 256, 256, 3), np.float32)
        else:
            images = np.zeros((len(images_list), self.img_rows, self.img_cols, 1), np.float32)

        masks = np.zeros((len(images_list), self.img_rows, self.img_cols, 1), np.float32)

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
                image = cv2.imread(image_dir + os.sep + image_name[0], cv2.IMREAD_GRAYSCALE)
                try:
                    image = cv2.resize(image, (self.img_rows, self.img_cols))
                except:
                    print(image_dir)
                    print(image_name[0])
                image = image[:, :, np.newaxis]
            images[i] = image / 127.5 - 1

            mask = cv2.imread(mask_dir + os.sep + image_name[0], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (self.img_rows, self.img_cols))
            mask = mask[:, :, np.newaxis]
            masks[i] = mask / 127.5 - 1

        return images, masks

    def train(self, samples, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        x_train, y_train = self.load_images(samples)
        x_train_vgg, _ = self.load_images(samples, vgg_preprocessing=True)

        # Adversarial loss ground truths
        global_valid = np.ones((batch_size,) + self.global_disc_patch)
        global_fake = np.zeros((batch_size,) + self.global_disc_patch)
        local_valid = np.ones((batch_size,) + self.local_disc_patch)
        local_fake = np.zeros((batch_size,) + self.local_disc_patch)

        g_train_interval = 1
        d_train_interval = 3
        save_interval = 1000

        # # pretrain
        # self.generator.load_weights(cfg.save_path + os.sep + cfg.experiment_name +
        #                             '/saved_model/generator_1000_weights.hdf5')
        # self.local_discriminator.load_weights(cfg.save_path + os.sep + cfg.experiment_name +
        #                             '/saved_model/l_discriminator_1000_weights.hdf5')
        # self.global_discriminator.load_weights(cfg.save_path + os.sep + cfg.experiment_name +
        #                             '/saved_model/g_discriminator_1000_weights.hdf5')

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            imgs = x_train[idx]
            vgg_imgs = x_train_vgg[idx]
            masks = y_train[idx]



            # Generate a batch of new images
            [gen_images1, gen_images] = self.generator.predict(masks)

            # local region
            crop_gen_images = gen_images[:, 40:88, 40:88, :]
            crop_images = imgs[:, 40:88, 40:88, :]
            crop_masks = masks[:, 40:88, 40:88, :]

            # update discriminator
            if (epoch == 0) or (epoch % d_train_interval == 0):

                # Train the local discriminator
                d_local_loss_real = self.local_discriminator.train_on_batch([crop_images, crop_masks], local_valid)
                d_local_loss_fake = self.local_discriminator.train_on_batch([crop_gen_images, crop_masks], local_fake)

                # Train the global discriminator
                d_global_loss_real = self.global_discriminator.train_on_batch([imgs, masks], global_valid)
                d_global_loss_fake = self.global_discriminator.train_on_batch([gen_images, masks], global_fake)

                d_local_loss = 0.5 * np.add(d_local_loss_real, d_local_loss_fake)
                d_global_loss = 0.5 * np.add(d_global_loss_real, d_global_loss_fake)
                d_loss = 0.5 * d_local_loss + 0.5 * d_global_loss

            # ---------------------
            #  Train Generator
            # ---------------------

            if (epoch == 0) or (epoch % g_train_interval == 0):
                # Extract ground truth image features using pre-trained VGG19 model
                image_features1, image_features2, image_features3, image_features4 = self.vgg.predict(vgg_imgs)

                # w_map * gen
                # w_imgs = self.gm_weight(imgs) * imgs
                mag_imgs = self.image_gradient_magnitude(gen_images)

                g_loss = self.combined.train_on_batch([imgs, masks], [local_valid,
                                                                      global_valid,
                                                                      imgs,
                                                                      mag_imgs,
                                                                      image_features1,
                                                                      image_features2,
                                                                      image_features3,
                                                                      image_features4])

            # Plot the progress
            print("[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f]" % (epoch, epochs, d_loss[0], 100*d_loss[1], np.sum(g_loss[0])))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                idx = np.random.randint(0, x_train.shape[0], 6)
                test_images = x_train[idx]
                test_masks = y_train[idx]
                self.sample_images(epoch, test_images, test_masks)

            if epoch % save_interval == 0:
                self.save_model(epoch)

    def sample_images(self, epoch, test_images, test_masks):
        r, c = 4, 6

        [gen_missing1, gen_missing] = self.generator.predict(test_masks)

        imgs = 0.5 * test_images + 0.5
        masked_imgs = 0.5 * test_masks + 0.5
        gen_missing = 0.5 * gen_missing + 0.5

        fig, axs = plt.subplots(r, c)
        for i in range(c):
            axs[0, i].imshow(imgs[i, :, :], cmap='gray')
            axs[0, i].axis('off')
            axs[1, i].imshow(masked_imgs[i], cmap='gray')
            axs[1, i].axis('off')
            axs[2, i].imshow(gen_missing1[i], cmap='gray')
            axs[2, i].axis('off')
            axs[3, i].imshow(gen_missing[i], cmap='gray')
            axs[3, i].axis('off')
        fig.savefig(cfg.save_path + os.sep + cfg.experiment_name + '/samples/%d.png' % epoch)
        plt.close()

    def save_model(self, epoch_number):

        def save(model, model_name):
            model_path = cfg.save_path + os.sep + cfg.experiment_name  + '/saved_model/%s.json' % model_name
            weights_path = cfg.save_path + os.sep + cfg.experiment_name  + '/saved_model/%s_weights.hdf5' % model_name
            options = {'file_arch': model_path,
                       'file_weight': weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, 'generator_' + str(epoch_number))
        save(self.global_discriminator, 'g_discriminator_' + str(epoch_number))
        save(self.local_discriminator, 'l_discriminator_' + str(epoch_number))


if __name__ == '__main__':
    import os

    data_list1 = os.listdir(cfg.dataset_path['sc_mask'])
    data_list1 = [[data, '1'] for data in data_list1]
    data_list2 = os.listdir(cfg.dataset_path['sf_mask'])
    data_list2 = [[data, '0'] for data in data_list2]

    # data spliting function
    def split_train_test(samples, VAL_SPLIT):
        split_idx = int(round(len(samples) * VAL_SPLIT))  # split index
        test = samples[:split_idx]
        train = samples[split_idx:]
        return train, test

    # train/test split (sc)
    sc_train_samples, sc_test_samples = split_train_test(data_list1, VAL_SPLIT=0.2)

    # train/test split (sf)
    sf_train_samples, sf_test_samples = split_train_test(data_list2, VAL_SPLIT=0.2)

    train_samples = sc_train_samples + sf_train_samples #
    test_samples = sc_test_samples + sf_test_samples #

    try:
        os.makedirs(cfg.save_path + os.sep + cfg.experiment_name)
        os.mkdir(cfg.save_path + os.sep + cfg.experiment_name + os.sep + 'samples')
        os.mkdir(cfg.save_path + os.sep + cfg.experiment_name + os.sep + 'saved_model')
        os.mkdir(cfg.save_path + os.sep + cfg.experiment_name + os.sep + 'test_results')
    except FileExistsError:
        print('folder exist')

    context_encoder = DilatedNetGANs()
    context_encoder.train(samples=train_samples,
                          epochs=cfg.training_params['epoch'],
                          batch_size=cfg.training_params['batch_size'],
                          sample_interval=cfg.training_params['sample_interval'])
