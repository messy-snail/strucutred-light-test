import glob
import cv2
import numpy as np
import copy
import math
import common as com
import yaml

class structured_light_system:
    def __init__(self):
        # self.black_threshold = 40
        self.black_threshold = 20
        # self.black_threshold = 50
        self.white_threshold = 5
        self.PATTERN_WIDTH = 0
        self.PATTERN_HEIGHT = 0

        self.K1 = None
        self.DIST1 = None

        self.K2 = None
        self.DIST2 = None

        self.Rt =None
        self.T = None

        self.DISTANCE = None



    def read_parameter(self, name):
        fs = cv2.FileStorage(name, cv2.FILE_STORAGE_READ)
        self.K1 = fs.getNode("cam_K").mat()
        self.DIST1 = fs.getNode("cam_kc").mat()

        self.K2 = fs.getNode("proj_K").mat()
        self.DIST2 = fs.getNode("proj_kc").mat()

        self.Rt = fs.getNode("R").mat().transpose()
        self.T = fs.getNode("T").mat()

    # def line_extarct(self):
    #     left_imgs, right_imgs, white_imgs, black_imgs =com.read_captured_imgs()
    #
    #     left_shadow_mask, right_shadow_mask = self.__compute_shadow_mask(white_imgs, black_imgs)
    #     cv2.imwrite('left_shadow_mask.png', left_shadow_mask)
    #     cv2.imwrite('right_shadow_mask.png', right_shadow_mask)
    #
    #     dec_xdir_img, dec_ydir_img = self.__get_projection_pixel(left_imgs, left_shadow_mask)
    #     # scale_x =1
    #     # scale_y = 1 if (self.PATTERN_WIDTH>self.PATTERN_HEIGHT) else 2
    #
    #     # proj_x = np.where((dec_xdir_img >= 0) & (dec_xdir_img < self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img < self.PATTERN_HEIGHT), dec_xdir_img, -1000)
    #     # proj_y = np.where((dec_xdir_img >= 0) & (dec_xdir_img < self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img < self.PATTERN_HEIGHT), dec_ydir_img, -1000)
    #     # proj_x_norm_img = cv2.normalize(proj_x, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #     # proj_x_color_img = cv2.applyColorMap(proj_x_norm_img, cv2.COLORMAP_JET)
    #     # proj_y_norm_img = cv2.normalize(proj_y, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    #     # proj_y_color_img = cv2.applyColorMap(proj_y_norm_img, cv2.COLORMAP_JET)
    #
    #     cols = dec_xdir_img.shape[1]
    #     rows = dec_xdir_img.shape[0]
    #     proj_x = np.where((dec_xdir_img >= 0) & (dec_xdir_img <= self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img <= self.PATTERN_HEIGHT), dec_xdir_img, -1)
    #     proj_y = np.where((dec_xdir_img >= 0) & (dec_xdir_img <= self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img <= self.PATTERN_HEIGHT), dec_ydir_img, -1)
    #
    #     inliers = np.where((proj_x.flatten()>=0) & (proj_y.flatten()>=0))
    #
    #     from collections import defaultdict
    #     cam_pts = defaultdict(list)
    #     pro_pts = defaultdict(list)
    #     index_dict = defaultdict(int)
    #
    #     for inlier in inliers[0]:
    #         index = int(proj_y[int(inlier/cols), inlier%cols]*self.PATTERN_WIDTH +proj_x[int(inlier/cols), inlier%cols])
    #         index_dict[index]+=1
    #         cam_pts[index]+=np.array([inlier%cols, int(inlier/cols)])
    #         pro_pts[index] = [proj_x[int(inlier / cols), inlier % cols], proj_y[int(inlier / cols), inlier % cols]]
    #
    #     self.__triangulate_stereo(cam_pts, pro_pts, 50)
    #
    #
    #
    #     # int scale_factor_x = 1;
    #     #     int scale_factor_y = (projector_size.width>projector_size.height ? 1 : 2);
    #     print('berry_test')

    def line_extarct(self):
        left_imgs, right_imgs, white_imgs, black_imgs =com.read_captured_imgs()

        left_shadow_mask, right_shadow_mask = self.__compute_shadow_mask(white_imgs, black_imgs)
        cv2.imwrite('left_shadow_mask.png', left_shadow_mask)
        cv2.imwrite('right_shadow_mask.png', right_shadow_mask)

        dec_xdir_img, dec_ydir_img = self.__get_projection_pixel(left_imgs, left_shadow_mask)
        # scale_x =1
        # scale_y = 1 if (self.PATTERN_WIDTH>self.PATTERN_HEIGHT) else 2

        # proj_x = np.where((dec_xdir_img >= 0) & (dec_xdir_img < self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img < self.PATTERN_HEIGHT), dec_xdir_img, -1000)
        # proj_y = np.where((dec_xdir_img >= 0) & (dec_xdir_img < self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img < self.PATTERN_HEIGHT), dec_ydir_img, -1000)
        # proj_x_norm_img = cv2.normalize(proj_x, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # proj_x_color_img = cv2.applyColorMap(proj_x_norm_img, cv2.COLORMAP_JET)
        # proj_y_norm_img = cv2.normalize(proj_y, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # proj_y_color_img = cv2.applyColorMap(proj_y_norm_img, cv2.COLORMAP_JET)

        cols = dec_xdir_img.shape[1]
        rows = dec_xdir_img.shape[0]
        proj_x = np.where((dec_xdir_img >= 0) & (dec_xdir_img <= self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img <= self.PATTERN_HEIGHT), dec_xdir_img, -1)
        proj_y = np.where((dec_xdir_img >= 0) & (dec_xdir_img <= self.PATTERN_WIDTH) & (dec_ydir_img >= 0) & (dec_ydir_img <= self.PATTERN_HEIGHT), dec_ydir_img, -1)

        proj_x_flatten = proj_x.flatten()
        proj_y_flatten = proj_y.flatten()

        inliers = np.where((proj_x_flatten >= 0) & (proj_y_flatten >= 0))


        max_size = self.PATTERN_WIDTH*self.PATTERN_HEIGHT



        from collections import defaultdict
        pro_pts = defaultdict(list)
        cam_avg_pts = defaultdict(list)

        for inlier in inliers[0]:
            x_val = proj_x_flatten[inlier]
            y_val = proj_y_flatten[inlier]

            if pro_pts[str([x_val, y_val])]==[]:
                pro_pts[str([x_val, y_val])]=[x_val, y_val]
                target = np.where((proj_x==x_val) & (proj_y==y_val))
                cam_avg_pts[str([x_val, y_val])] = [np.average(target[0]), np.average(target[1])]


            # if [x_val, y_val] not in pro_pts:
            #     pro_pts[index] = [x_val, y_val]
            #     index+=1
                # target = np.where((proj_x==x_val) & (proj_y==y_val))
                # cam_avg_pts = np.append(cam_avg_pts, np.array([[np.average(target[0]), np.average(target[1])]]), axis=0)


        index_pair = zip(range(0,self.PATTERN_WIDTH), range(0,self.PATTERN_HEIGHT))

        print('pass')



    def __triangulate_stereo(self, cam, proj, distance):

        undist_cam_pts = cv2.undistortPoints(cam, self.K1, self.DIST1)
        undist_proj_pts = cv2.undistortPoints(proj, self.K2, self.DIST2)


        pass
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
            img2 = np.where((shadow_mask == 255) & (even_img > odd_img), 255, 0)

            #-1=error
            img = np.where((shadow_mask ==255) & (abs(even_img-odd_img)< self.white_threshold), -1, img)
            img2 = np.where((shadow_mask ==255) & (abs(even_img-odd_img)< self.white_threshold), 0, img2)
            #-2=shadow인 곳
            img = np.where(shadow_mask == 0, -2, img)
            img2 = np.where(shadow_mask == 0, 20, img2)

            norm_img = cv2.normalize(img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
            color_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)

            # gray_for_viz_list.append(norm_img)
            img2_color = cv2.cvtColor(img2.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            img2_color[np.where(img2==20)] = (0,0,255)
            gray_for_viz_list.append(img2)
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
        dec += np.where(gray_img_list[0]==1, 2**(time_size-1), 0)

        for index in range(1, time_size):
            tmp = np.where(tmp>=0, tmp^gray_img_list[index], tmp)
            # tmp = tmp^gray_img_list[index]
            dec += np.where(tmp == 1, 2 ** (time_size - index - 1), 0)

        normzalied_dec = cv2.normalize(dec, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        color_dec = cv2.applyColorMap(normzalied_dec, cv2.COLORMAP_JET)

        return dec

    def decode(self):
        pass