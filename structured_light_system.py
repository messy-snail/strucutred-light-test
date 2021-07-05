import glob
import cv2
import numpy as np
import copy
import math
import common as com

class structured_light_system:
    def __init__(self):
        self.black_threshold = 40
        self.white_threshold = 5

    def line_extarct(self):
        left_imgs, right_imgs, white_imgs, black_imgs =com.read_captured_imgs()

        left_shadow_mask, right_shadow_mask = self.__compute_shadow_mask(white_imgs, black_imgs)
        cv2.imwrite('left_shadow_mask.png', left_shadow_mask)
        cv2.imwrite('right_shadow_mask.png', right_shadow_mask)

        dec_img1, dec_img2 = self.__get_projection_pixel(left_imgs, left_shadow_mask)
        print('berry_test')

    def __compute_shadow_mask(self, whites, blacks):
        shadow_maks_list = []
        for index in range(0, len(whites)):
            shadow_mask = np.where(abs(whites[index] - blacks[index])>self.black_threshold, 255, 0)
            shadow_maks_list.append(shadow_mask.astype(np.uint8))

        return shadow_maks_list

    def __get_projection_pixel(self, pattern_imgs, shadow_mask):
        original_imgs = copy.deepcopy(pattern_imgs)
        gray_img_list = []
        gray_for_viz_list = []
        gray_for_viz_color_list = []

        col_imgs = math.ceil(math.log(com.PATTERN_WIDTH) / math.log(2))
        row_imgs = math.ceil(math.log(com.PATTERN_HEIGHT) / math.log(2))

        for index in range(0, col_imgs+row_imgs):
            even_img = original_imgs[index*2]
            odd_img = original_imgs[index * 2+1]
            img = np.where((shadow_mask ==255) & (even_img > odd_img), 1, 0)

            #-1=error
            img = np.where((shadow_mask ==255) & (abs(even_img-odd_img)< self.white_threshold), -1, img)
            #-2=shadow인 곳
            img = np.where(shadow_mask == 0, -2, img)

            norm_img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
            gray_for_viz_list.append(norm_img)
            gray_for_viz_color_list.append(color_img)
            gray_img_list.append(img)

        number = 0
        for color in gray_for_viz_color_list:
            cv2.imwrite(f'./color_img/{number:03d}.png', color)
            number+=1
        number=0
        for gray in gray_for_viz_list:
            cv2.imwrite(f'./gray_img/{number:03d}.png', gray)
            number+=1
        dec_img1 = self.__gray_to_deciaml_num(gray_img_list[:col_imgs])
        dec_img2 = self.__gray_to_deciaml_num(gray_img_list[col_imgs:])
        return dec_img1, dec_img2

    def __gray_to_deciaml_num(self, gray_img_list):
        dec = np.zeros(gray_img_list[0].shape)
        time_size = len(gray_img_list)

        tmp = gray_img_list[0]

        #처음에만
        dec += np.where(gray_img_list[0]==255, 2**(time_size-1), 0)

        for index in range(1, time_size):
            tmp = tmp^gray_img_list[index]
            dec += np.where(tmp == 1, 2 ** (time_size - index - 1), 0)

        normzalied_dec = cv2.normalize(dec, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        color_dec = cv2.applyColorMap(normzalied_dec, cv2.COLORMAP_JET)

        return dec