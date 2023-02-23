from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Concatenate, Activation, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential, Model

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import imutils

DATA_PATH = ''
K_STONE_DIR = 'data/new_dataset/kidney_stones/'
U_STONE_DIR = 'data/new_dataset/ureteral_stones/'
B_STONE_DIR = 'data/new_dataset/bladder_stones/'

INPUT_SIZE = 128


def build_generator():
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

    def dilated_net(d0, gf):
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

    channels = 1
    img_shape = (INPUT_SIZE, INPUT_SIZE, 1)

    # Image input
    d0_1 = Input(shape=img_shape)

    d0_2 = dilated_net(d0_1, 64)
    u0_1 = Conv2D(channels, kernel_size=3, strides=1, padding='same', activation='tanh')(d0_2)

    u1_2 = dilated_net(u0_1, 64)
    u0_2 = Conv2D(channels, kernel_size=3, strides=1, padding='same', activation='tanh')(u1_2)

    return Model(d0_1, [u0_1, u0_2])


def preprocessing(image):
    image = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE))
    image = image / 127.5 - 1
    image = image[np.newaxis, :, :, np.newaxis]

    return image


def postprocessing(image, output_size):
    image = np.squeeze(image)
    image = cv2.resize(image, (output_size, output_size))
    image = (image + 1) * 127.5

    return image


def random_augment(image):
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel)

    # image rotation
    angle = random.randrange(-5, 5)
    augmented_image = imutils.rotate(image, angle)

    # image horizontal flip
    if bool(random.getrandbits(1)):  # random flip/not flip
        augmented_image = cv2.flip(augmented_image, 1)

    # image vertical flip
    if bool(random.getrandbits(1)):
        augmented_image = cv2.flip(augmented_image, 0)  # image

    return augmented_image


def imerode(image, size=10):

    kernel = np.ones((size, size), np.uint8)
    image[:, :, 0] = cv2.erode(image[:, :, 0], kernel)
    image[:, :, 1] = cv2.erode(image[:, :, 1], kernel)
    image[:, :, 2] = cv2.erode(image[:, :, 2], kernel)

    return image


def stone_inpainting(model, full_image, full_gt, cropped_mask, contour_params, stone_type, display_image1, display_image2, save_name):
    from matplotlib.colors import NoNorm

    xm = contour_params[0]
    ym = contour_params[1]
    wm = contour_params[2]

    # randomly augment
    cropped_mask = random_augment(cropped_mask)

    # inner mask box
    temp = full_image.copy()
    temp2 = full_image.copy()
    temp[ym:ym + wm, xm:xm + wm] = cropped_mask
    # print(cropped_mask.shape[0], cropped_mask.shape[1])

    # crop outer box
    cropped_stone_with_mask = temp[ym - wm:ym + 2*wm, xm - wm:xm + 2*wm]
    cropped_org = temp2[ym - wm:ym + 2 * wm, xm - wm:xm + 2 * wm]

    # cv2.imshow('ip', cropped_org)
    # cv2.waitKey(0)
    #
    # cv2.imshow('ip', cropped_stone_with_mask)
    # cv2.waitKey(0)

    cropped_stone_with_mask = preprocessing(cropped_stone_with_mask)
    [_, gen_patch] = model.predict(cropped_stone_with_mask)
    gen_patch = postprocessing(gen_patch, output_size=3*wm)
    cropped_stone_with_mask = postprocessing(cropped_stone_with_mask, output_size=3 * wm)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(cropped_org/255, cmap='gray', norm=NoNorm())
    axs[0].axis('off')
    axs[1].imshow(cropped_stone_with_mask/255, cmap='gray', norm=NoNorm())
    axs[1].axis('off')
    axs[2].imshow(gen_patch/255, cmap='gray', norm=NoNorm())
    axs[2].axis('off')
        # plt.show()
    fig.savefig('inpaint_samples/sf_inpaint/' + save_name)
    plt.close()

    # print(xm, ym, wm, gen_patch.shape[0], gen_patch.shape[1])

    # replace back at the same position
    full_image[ym - wm:ym + 2*wm, xm - wm:xm + 2*wm] = gen_patch

    full_gt[ym:ym + wm, xm:xm + wm] = cropped_mask

    # display image
    display_image2[ym - wm:ym + 2*wm, xm - wm:xm + 2*wm, 0] = gen_patch
    display_image2[ym - wm:ym + 2*wm, xm - wm:xm + 2*wm, 1] = gen_patch
    display_image2[ym - wm:ym + 2*wm, xm - wm:xm + 2*wm, 2] = gen_patch

    if stone_type == 2:  # k stone
        cv2.rectangle(display_image1, (xm, ym), (xm + wm, ym + wm), (0, 0, 255), 1)
        cv2.rectangle(display_image2, (xm, ym), (xm + wm, ym + wm), (0, 0, 255), 1)
    elif stone_type == 1:  # u stone
        cv2.rectangle(display_image1, (xm, ym), (xm + wm, ym + wm), (0, 255, 0), 1)
        cv2.rectangle(display_image2, (xm, ym), (xm + wm, ym + wm), (0, 255, 0), 1)
    elif stone_type == 0:  # b stone
        cv2.rectangle(display_image1, (xm, ym), (xm + wm, ym + wm), (255, 0, 0), 1)
        cv2.rectangle(display_image2, (xm, ym), (xm + wm, ym + wm), (255, 0, 0), 1)

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(display_image1, cmap='gray')
    # axs[0].axis('off')
    # axs[1].imshow(display_image2, cmap='gray')
    # axs[1].axis('off')
    # plt.show()

    return full_image, full_gt, display_image1, display_image2


def stone_gen(sf_list):
    import random

    # Build the generator
    generator = build_generator()
    generator.summary()
    generator.load_weights('saved_model/generator(1-10-1)_10000_weights.hdf5')

    # stone list from cropped stone databased
    k_stone_list = os.listdir(K_STONE_DIR)
    u_stone_list = os.listdir(U_STONE_DIR)
    b_stone_list = os.listdir(B_STONE_DIR)

    for i, image_name in enumerate(sf_list):
        # org image
        org = cv2.imread(DATA_PATH + 'all_images_full' + os.sep + image_name[0], cv2.IMREAD_GRAYSCALE)
        # org = cv2.resize(org, (round(org.shape[1] / (org.shape[0] / 1024)), 1024))   ###

        # stones gt
        # gt = cv2.imread(DATA_PATH + 'all_groundtruth' + os.sep + image_name[1], cv2.IMREAD_GRAYSCALE)

        # kub map
        kub_map = cv2.imread(DATA_PATH + 'Full_KUB_map' + os.sep + image_name[0][:-4] + '.png')
        kub_map = cv2.resize(kub_map, (org.shape[1], org.shape[0]))
        kub_map = imerode(kub_map)

        # shifting
        translation_matrix = np.float32([[1, 0, 0], [0, 1, -12]])
        kub_map = cv2.warpAffine(kub_map, translation_matrix, (org.shape[1], org.shape[0]))

        min_width = 7 * 3
        max_width = 35 * 3   #22

        augment_per_img = 10
        max_stones = 3
        random_insert = True
        for num_img in range(augment_per_img):

            gen_image = org.copy()
            # gen_gt = gt.copy()   # np.zeros((gen_image.shape[0], gen_image.shape[1]), dtype=np.uint8)
            gen_gt = np.zeros((gen_image.shape[0], gen_image.shape[1]), dtype=np.uint8)

            # display
            display_image1 = np.zeros((gen_image.shape[0], gen_image.shape[1], 3), dtype=np.uint8)
            display_image1[:, :, 0] = gen_image
            display_image1[:, :, 1] = gen_image
            display_image1[:, :, 2] = gen_image
            display_image2 = np.zeros((gen_image.shape[0], gen_image.shape[1], 3), dtype=np.uint8)
            display_image2[:, :, 0] = gen_image
            display_image2[:, :, 1] = gen_image
            display_image2[:, :, 2] = gen_image

            if random_insert:
                # if random.getrandbits:
                #     kub_map[:, :, 2] = 0
                #     max_stones = 2
                location_list = np.argwhere(kub_map == 255)           # get all location map coordinate list
                stone_per_img = random.randint(1, max_stones)
                num_stone = 1
                while num_stone <= stone_per_img:
                    # random select stones
                    location = random.choice(location_list)

                    # find stone type
                    if location[2] == 2:  # kidney stone
                        stone_name = random.choice(k_stone_list)
                        stone_mask = cv2.imread((K_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

                    elif location[2] == 1:  # ureter stone
                        stone_name = random.choice(u_stone_list)
                        stone_mask = cv2.imread((U_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

                    elif location[2] == 0:  # bladder region
                        stone_name = random.choice(b_stone_list)
                        stone_mask = cv2.imread((B_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

                    # check stone's properties
                    stone_mask = cv2.resize(stone_mask, (round(stone_mask.shape[1] * 3), round(stone_mask.shape[1] * 3)))

                    contour_params = [location[1], location[0], stone_mask.shape[0], stone_mask.shape[1]]
                    if min_width < stone_mask.shape[0] < max_width:
                        try:
                            save_name = image_name[0][:-4] + '_' + str(num_img) + '.png'
                            gen_image, gen_gt, save_org, save_input = stone_inpainting(generator,
                                                                                       gen_image,
                                                                                       gen_gt,
                                                                                       stone_mask,
                                                                                       contour_params,
                                                                                       location[2],
                                                                                       display_image1,
                                                                                       display_image2,
                                                                                       save_name)
                            num_stone += 1
                            # save_img = np.concatenate((save_org, save_input, save_output), axis=1)
                            # cv2.imwrite('inpainted_sf_results/cropped_sf/' + image_name[0][:-4] + '_' + str(num_stone) + '.png', save_org)
                            # cv2.imwrite('inpainted_sf_results/cropped_sf_with_mask/' + image_name[0][:-4] + '_' + str(
                            #     num_stone) + '.png', save_input)
                            # cv2.imwrite('inpainted_sf_results/results/' + image_name[0][:-4] + '_' + str(num_stone) + '.png', save_img)

                        except:
                            print(image_name[0][:-4])

                # gen_image_location = np.concatenate((display_image1, display_image2), axis=1)

                # gen_image = cv2.resize(gen_image, (round(org.shape[1] / (org.shape[0] / 1024)), 1024))
                # gen_gt = cv2.resize(gen_gt, (gen_image.shape[1], gen_image.shape[0]))
                #
                # cv2.imwrite('data/1-10-1/all_size/g(sf)_images' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png', gen_image)
                # cv2.imwrite('data/1-10-1/all_size/g(sf)_gt' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png', gen_gt)
                #
                # if num_img < 3:
                #     cv2.imwrite('data/G(sc)_images_location' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png',
                #                 gen_image_location)

            else:

                # kidney stone
                num_k_stone = 0
                while num_k_stone < 1:
                    location = random.choice(np.argwhere(kub_map[:, :, 2] == 255))
                    stone_name = random.choice(k_stone_list)
                    stone_mask = cv2.imread((K_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

                    # check stone's properties
                    contour_params = [location[1], location[0], stone_mask.shape[0], stone_mask.shape[1]]
                    if min_width < stone_mask.shape[0] < max_width:
                        try:
                            gen_image, gen_gt, display_image1, display_image2 = stone_inpainting(generator,
                                                                                                 gen_image,
                                                                                                 gen_gt,
                                                                                                 stone_mask,
                                                                                                 contour_params,
                                                                                                 2,
                                                                                                 display_image1,
                                                                                                 display_image2)
                            num_k_stone += 1
                        except:
                            print(image_name[0][:-4])

                # ureter stone
                num_u_stone = 0
                while num_u_stone < 1:
                    location = random.choice(np.argwhere(kub_map[:, :, 1] == 255))
                    stone_name = random.choice(u_stone_list)
                    stone_mask = cv2.imread((U_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

                    # check stone's properties
                    contour_params = [location[1], location[0], stone_mask.shape[0], stone_mask.shape[1]]
                    if min_width < stone_mask.shape[0] < max_width:
                        try:
                            gen_image, gen_gt, display_image1, display_image2 = stone_inpainting(generator,
                                                                                                 gen_image,
                                                                                                 gen_gt,
                                                                                                 stone_mask,
                                                                                                 contour_params,
                                                                                                 1,
                                                                                                 display_image1,
                                                                                                 display_image2)
                            num_u_stone += 1
                        except:
                            print(image_name[0][:-4])

                # bladder region
                num_b_stone = 0
                while num_b_stone < 1:
                    location = random.choice(np.argwhere(kub_map[:, :, 0] == 255))
                    stone_name = random.choice(b_stone_list)
                    stone_mask = cv2.imread((B_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

                    # check stone's properties
                    contour_params = [location[1], location[0], stone_mask.shape[0], stone_mask.shape[1]]
                    if min_width < stone_mask.shape[0] < max_width:
                        try:
                            gen_image, gen_gt, display_image1, display_image2 = stone_inpainting(generator,
                                                                                                 gen_image,
                                                                                                 gen_gt,
                                                                                                 stone_mask,
                                                                                                 contour_params,
                                                                                                 0,
                                                                                                 display_image1,
                                                                                                 display_image2)
                            num_b_stone += 1
                        except:
                            print(image_name[0][:-4])

                gen_image_location = np.concatenate((display_image1, display_image2), axis=1)
                cv2.imwrite('data/G(sf)_images' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png',
                            gen_image)
                cv2.imwrite('data/G(sf)_gt' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png',
                            gen_gt)
                # cv2.imwrite('data/G(sf)_images_location' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png',
                #             gen_image_location)


def stone_gen_sc(sc_list):
    import random

    # Build the generator
    generator = build_generator()
    generator.summary()
    generator.load_weights('saved_model/generator(1-10-1)_10000_weights.hdf5')

    # stone list from cropped stone databased
    k_stone_list = os.listdir(K_STONE_DIR)
    u_stone_list = os.listdir(U_STONE_DIR)
    b_stone_list = os.listdir(B_STONE_DIR)

    for i, image_name in enumerate(sc_list):
        # org image
        org = cv2.imread(DATA_PATH + 'all_images_full' + os.sep + image_name[0], cv2.IMREAD_GRAYSCALE)
        org = cv2.resize(org, (round(org.shape[1] / (org.shape[0] / 1024)), 1024))

        # stones gt
        gt = cv2.imread(DATA_PATH + 'all_groundtruth' + os.sep + image_name[1], cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (org.shape[1], org.shape[0]))

        # kub map
        kub_map = cv2.imread(DATA_PATH + 'Full_KUB_map' + os.sep + image_name[0][:-4] + '.png')
        kub_map = cv2.resize(kub_map, (org.shape[1], org.shape[0]))
        kub_map = imerode(kub_map)

        # shifting
        translation_matrix = np.float32([[1, 0, 0], [0, 1, -15]])
        kub_map = cv2.warpAffine(kub_map, translation_matrix, (org.shape[1], org.shape[0]))

        min_width = 7
        max_width = 35  #25

        augment_per_img = 10
        flag_save = False

        # loop for generating augmented images
        for num_img in range(augment_per_img):

            gen_image = org.copy()
            gen_gt = gt.copy()

            # display
            display_image1 = np.zeros((gen_image.shape[0], gen_image.shape[1], 3), dtype=np.uint8)
            display_image1[:, :, 0] = gen_image
            display_image1[:, :, 1] = gen_image
            display_image1[:, :, 2] = gen_image
            display_image2 = np.zeros((gen_image.shape[0], gen_image.shape[1], 3), dtype=np.uint8)
            display_image2[:, :, 0] = gen_image
            display_image2[:, :, 1] = gen_image
            display_image2[:, :, 2] = gen_image

            # find cc in img
            cnt_tmp = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            stone_cc = cnt_tmp[0] if len(cnt_tmp) == 2 else cnt_tmp[1]
            total_cc = len(stone_cc)
            # print(total_cc)

            # loop stone cc
            erase_count = 0
            for i, cc in enumerate(stone_cc):

                # tmp_cc = np.zeros(gt.shape, np.uint8)
                # cv2.drawContours(tmp_cc, [cc], -1, 255, -1)

                # stone properties
                x_s, y_s, w_s, h_s = cv2.boundingRect(cc)
                w_m = np.max([w_s, h_s])
                # print(i, x_s, y_s, w_s, h_s)
                # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 3)

                # cv2.imshow('img', tmp_true_cnt)
                # cv2.waitKey(0)

                # randomly inpaint stone or background
                erase = False
                if total_cc == 1 or bool(random.getrandbits(1)):
                    # load stone based on location
                    if kub_map[y_s + round(h_s / 2), x_s + round(w_s / 2), 2] == 255:
                        stone_type = 2
                        stone_name = random.choice(k_stone_list)
                        stone_mask = cv2.imread((K_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
                    elif kub_map[y_s + round(h_s / 2), x_s + round(w_s / 2), 1] == 255:
                        stone_type = 1
                        stone_name = random.choice(u_stone_list)
                        stone_mask = cv2.imread((U_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
                    elif kub_map[y_s + round(h_s / 2), x_s + round(w_s/ 2), 0] == 255:
                        stone_type = 0
                        stone_name = random.choice(b_stone_list)
                        stone_mask = cv2.imread((B_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
                    else:
                        stone_type = 2
                        stone_name = random.choice(k_stone_list)
                        stone_mask = cv2.imread((K_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)

                else:
                    erase = True
                    erase_count += 1
                    stone_mask = np.zeros((w_m, w_m), np.uint8)

                stone_mask = cv2.resize(stone_mask, (w_m, w_m))
                x_t = x_s + round(w_s / 2) - round(w_m / 2)
                y_t = y_s + round(h_s / 2) - round(w_m / 2)

                # random real or inpainted
                if 1:

                    # check stone's properties
                    contour_params = [x_t, y_t, w_m, w_m]
                    if min_width < stone_mask.shape[0] < max_width:
                        flag_save = True
                        save_name = image_name[0][:-4] + '_' + str(num_img) + '.png'
                        try:
                            gen_image, gen_gt, save_org, save_input = stone_inpainting(generator,
                                                                                       gen_image,
                                                                                       gen_gt,
                                                                                       stone_mask,
                                                                                       contour_params,
                                                                                       stone_type,
                                                                                       display_image1,
                                                                                       display_image2,
                                                                                       save_name,
                                                                                       erase)

                        except:
                            print(image_name[0][:-4])

            # # compensate erased stones
            # location_list = np.argwhere(kub_map == 255)
            # num_stone = 0
            # # print('erase_count ', erase_count)
            # if erase_count > 3:
            #     erase_count = 3
            #
            # while num_stone < erase_count:
            #     # random select stones
            #     location = random.choice(location_list)
            #
            #     # find stone type
            #     if location[2] == 2:  # kidney stone
            #         stone_name = random.choice(k_stone_list)
            #         stone_mask = cv2.imread((K_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            #
            #     elif location[2] == 1:  # ureter stone
            #         stone_name = random.choice(u_stone_list)
            #         stone_mask = cv2.imread((U_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            #
            #     elif location[2] == 0:  # bladder region
            #         stone_name = random.choice(b_stone_list)
            #         stone_mask = cv2.imread((B_STONE_DIR + os.sep + stone_name), cv2.IMREAD_GRAYSCALE)
            #
            #     # check stone's properties
            #     contour_params = [location[1], location[0], stone_mask.shape[0], stone_mask.shape[1]]
            #     if min_width < stone_mask.shape[0] < max_width:
            #         try:
            #             gen_image, gen_gt, save_org, save_input = stone_inpainting(generator,
            #                                                                        gen_image,
            #                                                                        gen_gt,
            #                                                                        stone_mask,
            #                                                                        contour_params,
            #                                                                        location[2],
            #                                                                        display_image1,
            #                                                                        display_image2)
            #             num_stone += 1
            #
            #         except:
            #             print(image_name[0][:-4])

            # if flag_save:
                # gen_image_location = np.concatenate((display_image1, display_image2), axis=1)
                # cv2.imwrite('data/1-10-1/g(sc)_images' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png', gen_image)
                # cv2.imwrite('data/1-10-1/g(sc)_gt' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png', gen_gt)

                # if num_img < 3:
                #     cv2.imwrite('data/G(sc)_images_location' + os.sep + image_name[0][:-4] + '_' + str(num_img) + '.png',
                #                 gen_image_location)


if __name__ == '__main__':
    import pandas as pd

    # load full images
    excel = pd.read_excel('data/full_images/image_list_full.xlsx')
    images_list = pd.DataFrame(excel, columns=['image', 'gt', 'stone']).values.tolist()
    sc_list = images_list[:1156]
    sf_list = images_list[1156:2356]  # 1200 images  (stone-free)
    # sf_list = sf_list[:100]
    stone_gen(sf_list)
