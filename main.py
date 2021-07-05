import copy

from pyflycap2.interface import GUI
from pyflycap2.interface import Camera

from _thread import*

import numpy as np
import cv2
import glob
import open3d as o3d
import math

class main:
    def __init__(self):
        self.cam_serial1 = 0
        self.cam_serial2 = 0
        self.capture_name = ''

        self.PATTERN_WIDTH = 1280
        self.PATTERN_HEIGHT = 800

        # self.PATTERN_WIDTH = 640
        # self.PATTERN_HEIGHT = 480

        # self.PATTERN_WIDTH = 8
        # self.PATTERN_HEIGHT = 8

        self.EXPOSURE_TIME = 500

        self.CAPTURE_FLAG = False

        self.CAPTURE_INDEX = 0

        self.thread_run = False
        self.thread_run2 = False

        self.black_threshold = 40
        self.white_threshold = 5

    def cam_thread(self, id):
        self.capture_name1 = 'left'
        self.capture_name2 = 'right'

        c1 = Camera(serial=self.cam_serial1)
        # c2 = Camera(serial=self.cam_serial2)

        c1.connect()

        # c1.set_cam_setting_option_values('exposure',self.EXPOSURE_TIME)
        c1.start_capture()
        print('c1: ',c1.is_color)
        # c1.set_default_color_processing('NN')

        # c2.connect()
        # c2.set_cam_setting_option_values('exposure', self.EXPOSURE_TIME)
        # c2.start_capture()
        # print('c2: ', c2.is_color())
        while self.thread_run == True:
            cv2.waitKey(500)
            c1.read_next_image()
            image1 = c1.get_current_image()  # last image
            imageData1 = np.asarray(image1["buffer"], dtype=np.byte)
            cv_image1 = np.array(image1["buffer"], dtype="uint8").reshape((image1["rows"], image1["cols"]));
            cv_image1 = cv2.cvtColor(cv_image1, cv2.COLOR_BayerGB2BGR)
            cv_image1 = cv2.cvtColor(cv_image1, cv2.COLOR_BGR2GRAY)

            # c2.read_next_image()
            # image2 = c2.get_current_image()  # last image
            # imageData2 = np.asarray(image2["buffer"], dtype=np.byte)
            # cv_image2 = np.array(image2["buffer"], dtype="uint8").reshape((image1["rows"], image1["cols"]));

            cv2.imshow(f'left-{id}', cv_image1)
            # cv2.imshow(f'right-{id}', cv_image2)
            cv2.waitKey(10)

            if self.CAPTURE_FLAG:
                ret1 = False
                ret1 = cv2.imwrite(f'./captured/{self.capture_name1}-{self.CAPTURE_INDEX:02d}.png', cv_image1)
                # ret2 = False
                # ret2 = cv2.imwrite(f'./captured/{self.capture_name2}-{self.CAPTURE_INDEX:02d}.png', cv_image2)

                while True:
                    if ret1==True:
                        self.CAPTURE_FLAG = False
                        self.CAPTURE_INDEX+=1
                        break
        c1.disconnect()
        # c2.disconnect()
        self.CAPTURE_INDEX = 0
    def cam_thread2(self, id):
        self.capture_name1 = 'left'
        self.capture_name2 = 'right'

        c1 = None
        c2 = None
        try:
            c1 = Camera(serial=self.cam_serial1)
            c2 = Camera(serial=self.cam_serial2)
        except Exception as e:
            print('랜 연결 또는 무선 인터넷 연결 여부를 확인하세요')
            print(e)
            return

        c1.connect()
        c1.start_capture()

        c1.set_cam_setting_option_values('exposure', self.EXPOSURE_TIME)
        # print('c1: ', c1.is_color())

        c2.connect()
        c2.start_capture()
        c2.set_cam_setting_option_values('exposure', self.EXPOSURE_TIME)
        # print('c2: ', c2.is_color())

        while self.thread_run2 == True:
            cv2.waitKey(100)
            c1.read_next_image()
            image1 = c1.get_current_image()  # last image
            imageData1 = np.asarray(image1["buffer"], dtype=np.byte)
            cv_image1 = np.array(image1["buffer"], dtype="uint8").reshape((image1["rows"], image1["cols"]));

            c2.read_next_image()
            image2 = c2.get_current_image()  # last image
            imageData2 = np.asarray(image2["buffer"], dtype=np.byte)
            cv_image2 = np.array(image2["buffer"], dtype="uint8").reshape((image1["rows"], image1["cols"]));


            cv2.namedWindow("left", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("left", 640, 480)
            cv2.moveWindow("left", 0, 0)

            cv2.namedWindow("right", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("right", 640, 480)
            cv2.moveWindow("right", 642, 0)

            cv2.imshow('left', cv_image1)
            cv2.imshow('right', cv_image2)

        c1.disconnect()
        c2.disconnect()
        self.CAPTURE_INDEX = 0

    def cam_capture(self):

        self.cam_serial1 = 14193278
        # self.cam_serial2 = 14193278
        self.thread_run = True
        start_new_thread(self.cam_thread, (0,))
        gray_code_generator = cv2.structured_light.GrayCodePattern_create(self.PATTERN_WIDTH, self.PATTERN_HEIGHT)

        _, patter_images = gray_code_generator.generate()
        # white black 뿌림
        patter_images.append(255 * np.ones((self.PATTERN_HEIGHT, self.PATTERN_WIDTH)))
        patter_images.append(np.zeros((self.PATTERN_HEIGHT, self.PATTERN_WIDTH)))
        cv2.namedWindow("Pattern Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pattern Window", self.PATTERN_WIDTH, self.PATTERN_HEIGHT)
        cv2.moveWindow("Pattern Window", 1920, 0)
        cv2.setWindowProperty("Pattern Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        print(f'Pattern Size: {len(patter_images)}')
        num = 0
        for image in patter_images:
            cv2.imwrite(f"pattern/pattern-{num}.png", image)
            cv2.imwrite(f"captured/right-{num:02d}.png", image)
            num += 1

        for image in patter_images:
            cv2.imshow("Pattern Window", image)
            cv2.waitKey(10)
            self.CAPTURE_FLAG = True
            while self.CAPTURE_FLAG:
                continue

        self.thread_run = False
        return True
    def cam_stream(self):

        self.cam_serial1 = 13142459
        self.cam_serial2 = 14193278
        self.thread_run2 = True
        start_new_thread(self.cam_thread2, (0,))
        return True

    def structured_light_reconstruction(self, dataset):
        left_img_list = []
        right_img_list = []
        if dataset=='capture' or 'c':
            # left_img_list = glob.glob('./captured/left-*.png')
            # right_img_list = glob.glob('./captured/right-*.png')
            left_img_list = glob.glob('./captured/right-*.png')
            right_img_list = glob.glob('./captured/left-*.png')

        elif dataset=='back' or 'b':
            left_img_list = glob.glob('./statue_back_side/pattern_cam1_*.jpg')
            right_img_list = glob.glob('./statue_back_side/pattern_cam2_*.jpg')
        elif dataset=='front' or 'f':
            left_img_list = glob.glob('./statue_front_side/pattern_cam1_*.jpg')
            right_img_list = glob.glob('./statue_front_side/pattern_cam2_*.jpg')

        gray_code_generator = cv2.structured_light.GrayCodePattern_create(self.PATTERN_WIDTH, self.PATTERN_HEIGHT)

        # gray_code_generator.setWhiteThreshold(240)
        # gray_code_generator.setBlackThreshold(230)

        calib_file_name = './calibrationParameters.yml'
        fs = cv2.FileStorage(calib_file_name, cv2.FILE_STORAGE_READ)

        cam1intrinsics  = fs.getNode('cam1_intrinsics').mat()
        cam1distCoeffs = fs.getNode('cam1_distorsion').mat()

        cam2intrinsics = fs.getNode('cam2_intrinsics').mat()
        cam2distCoeffs = fs.getNode('cam2_distorsion').mat()

        R = fs.getNode('R').mat()
        T = fs.getNode('T').mat()

        img_size = (1280,960)
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cam1intrinsics, cam1distCoeffs, cam2intrinsics,
                                                                         cam2distCoeffs, img_size, R, T)

        # map1x, map1y, map2x, map2y;
        # map1x, map1y = cv2.initUndistortRectifyMap(cam1intrinsics, cam1distCoeffs, R1, P1, img_size, cv2.CV_32FC1)
        # map2x, map2y = cv2.initUndistortRectifyMap(cam2intrinsics, cam2distCoeffs, R2, P2, img_size, cv2.CV_32FC1)

        left_imgs = []
        right_imgs = []

        white_imgs = []
        black_imgs = []
        for index in range(0, len(left_img_list)):
            # cv2.remap
            if index<len(left_img_list)-2:
                left_imgs.append(cv2.imread(left_img_list[index], cv2.IMREAD_GRAYSCALE))
                right_imgs.append(cv2.imread(right_img_list[index], cv2.IMREAD_GRAYSCALE))
            elif index==len(left_img_list)-2:
                white_imgs.append(cv2.imread(left_img_list[index], cv2.IMREAD_GRAYSCALE))
                white_imgs.append(cv2.imread(right_img_list[index], cv2.IMREAD_GRAYSCALE))
            elif index==len(left_img_list)-1:
                black_imgs.append(cv2.imread(left_img_list[index], cv2.IMREAD_GRAYSCALE))
                black_imgs.append(cv2.imread(right_img_list[index], cv2.IMREAD_GRAYSCALE))


        ret, bin0 = cv2.threshold(white_imgs[0], 250, 255, cv2.THRESH_BINARY)
        ret, bin1 = cv2.threshold(white_imgs[1], 250, 255, cv2.THRESH_BINARY)

        ret, disparity = gray_code_generator.decode([left_imgs, right_imgs], None, white_imgs, black_imgs)
        print(left_img_list)
        print(right_img_list)

        normalized_disparity = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        color_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET);

        color_disparity = cv2.resize(color_disparity, (640, 480))
        cv2.imshow('disparity map', color_disparity)
        cv2.waitKey(10)

    def structured_light_reconstruction2(self, dataset):
        left_img_list = []
        right_img_list = []
        if dataset=='capture' or 'c':
            # left_img_list = glob.glob('./captured/left-*.png')
            # right_img_list = glob.glob('./captured/right-*.png')
            left_img_list = glob.glob('./captured/right-*.png')
            right_img_list = glob.glob('./captured/left-*.png')

        elif dataset=='back' or 'b':
            left_img_list = glob.glob('./statue_back_side/pattern_cam1_*.jpg')
            right_img_list = glob.glob('./statue_back_side/pattern_cam2_*.jpg')
        elif dataset=='front' or 'f':
            left_img_list = glob.glob('./statue_front_side/pattern_cam1_*.jpg')
            right_img_list = glob.glob('./statue_front_side/pattern_cam2_*.jpg')

        gray_code_generator = cv2.structured_light.GrayCodePattern_create(self.PATTERN_WIDTH, self.PATTERN_HEIGHT)

        # gray_code_generator.setWhiteThreshold(240)
        # gray_code_generator.setBlackThreshold(230)

        calib_file_name = './calibrationParameters.yml'
        fs = cv2.FileStorage(calib_file_name, cv2.FILE_STORAGE_READ)

        cam1intrinsics  = fs.getNode('cam1_intrinsics').mat()
        cam1distCoeffs = fs.getNode('cam1_distorsion').mat()

        cam2intrinsics = fs.getNode('cam2_intrinsics').mat()
        cam2distCoeffs = fs.getNode('cam2_distorsion').mat()

        R = fs.getNode('R').mat()
        T = fs.getNode('T').mat()

        img_size = (1280,960)
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cam1intrinsics, cam1distCoeffs, cam2intrinsics,
                                                                         cam2distCoeffs, img_size, R, T)

        # map1x, map1y, map2x, map2y;
        # map1x, map1y = cv2.initUndistortRectifyMap(cam1intrinsics, cam1distCoeffs, R1, P1, img_size, cv2.CV_32FC1)
        # map2x, map2y = cv2.initUndistortRectifyMap(cam2intrinsics, cam2distCoeffs, R2, P2, img_size, cv2.CV_32FC1)

        left_imgs = []
        right_imgs = []

        white_imgs = []
        black_imgs = []
        for index in range(0, len(left_img_list)):
            # cv2.remap
            if index<len(left_img_list)-2:
                left_imgs.append(cv2.imread(left_img_list[index], cv2.IMREAD_GRAYSCALE))
                right_imgs.append(cv2.imread(right_img_list[index], cv2.IMREAD_GRAYSCALE))
            elif index==len(left_img_list)-2:
                white_imgs.append(cv2.imread(left_img_list[index], cv2.IMREAD_GRAYSCALE))
                white_imgs.append(cv2.imread(right_img_list[index], cv2.IMREAD_GRAYSCALE))
            elif index==len(left_img_list)-1:
                black_imgs.append(cv2.imread(left_img_list[index], cv2.IMREAD_GRAYSCALE))
                black_imgs.append(cv2.imread(right_img_list[index], cv2.IMREAD_GRAYSCALE))


        ret, bin0 = cv2.threshold(white_imgs[0], 250, 255, cv2.THRESH_BINARY)
        ret, bin1 = cv2.threshold(white_imgs[1], 250, 255, cv2.THRESH_BINARY)

        ret, disparity = gray_code_generator.decode([left_imgs, right_imgs], None, white_imgs, black_imgs)
        print(left_img_list)
        print(right_img_list)

        normalized_disparity = cv2.normalize(disparity, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        color_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_JET);

        color_disparity = cv2.resize(color_disparity, (640, 480))
        cv2.imshow('disparity map', color_disparity)
        cv2.waitKey(10)

    def generate_pattern(self):
        gray_code_generator = cv2.structured_light.GrayCodePattern_create(self.PATTERN_WIDTH, self.PATTERN_HEIGHT)
        _, patter_images = gray_code_generator.generate()
        patter_images.append(255 * np.ones((self.PATTERN_HEIGHT, self.PATTERN_WIDTH)))
        patter_images.append(np.zeros((self.PATTERN_HEIGHT, self.PATTERN_WIDTH)))
        print(f'Pattern Size: {len(patter_images)}')
        num = 0
        for image in patter_images:
            cv2.imwrite(f"gray-row-col/pattern-{num}.png", image)
            num += 1

    def cam_test(self):
        cap = cv2.VideoCapture(0+cv2.CAP_DSHOW)

        if not cap.isOpened():
            print('fail')

        cap.set(cv2.CAP_PROP_SETTINGS,1 )
        while True:
            _, frame = cap.read()
            cv2.imshow('test', frame)
            key = cv2.waitKey(60)
            if key == 27:
                break

    def line_test(self):
        # target_img = cv2.imread('./captured/right-33.png', 0)[10:-490, 80:-420]
        # white = cv2.imread('./captured/right-40.png', 0)[10:-490, 80:-420]
        # black = cv2.imread('./captured/right-41.png', 0)[10:-490, 80:-420]

        target_img = cv2.imread('./captured/left-5.png', 0)
        white = cv2.imread('./captured/left-0.png', 0)
        black = cv2.imread('./statue_back_side/pattern_cam1_im44.jpg', 0)

        test_img = target_img - white
        cv2.normalize(test_img, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        threshold_val = 120
        diff_val = 0

        diff = abs(white.astype(np.int)-target_img.astype(np.int))
        diff_img = copy.deepcopy(target_img)
        diff_img[np.where(diff > 1)] = 255
        diff_img[np.where(diff <= 1)] = 0
        norm_diff = cv2.normalize(diff, None, 255, 0, cv2.NORM_MINMAX)
        diff_img = copy.deepcopy(target_img)
        diff_img[np.where(diff<0)] = 0
        # diff_img[np.where(diff == 0)] = 0

        # _, diff_th_img = cv2.threshold(diff_img, 10, 255, cv2.THRESH_BINARY)
        cv2.imwrite('diff_img_result.png', diff_img)
        and_operation = cv2.bitwise_and(white, target_img)

        _,white = cv2.threshold(white,threshold_val, 255, cv2.THRESH_BINARY)
        _,target_img = cv2.threshold(target_img, threshold_val, 255, cv2.THRESH_BINARY)
        and_operation_result = cv2.bitwise_and(white, target_img)
        cv2.imwrite('and_operation_result.png', and_operation_result)

        concat_img_test = cv2.hconcat([cv2.resize(diff_img, (640,480)), cv2.resize(and_operation_result, (640,480))])
        cv2.imshow('concat_img_test', concat_img_test)
        ret, bin_img = cv2.threshold(and_operation_result, threshold_val, 255, cv2.THRESH_BINARY)
        bin_img32 = cv2.Canny(bin_img, 200, 250)
        target_color = cv2.cvtColor(target_img, cv2.COLOR_GRAY2BGR)
        canny_overlay = copy.deepcopy(target_color)
        canny_overlay[np.where(bin_img32 == 255)] = [0, 0, 255]

        concat_img31 = cv2.hconcat([target_img, bin_img])
        concat_img32 = cv2.hconcat([cv2.cvtColor(bin_img32, cv2.COLOR_GRAY2BGR), canny_overlay])
        cv2.imwrite('final_image.png', cv2.vconcat([cv2.cvtColor(concat_img31, cv2.COLOR_GRAY2BGR), concat_img32]))
        cv2.imshow('final_image', cv2.vconcat([cv2.cvtColor(concat_img31, cv2.COLOR_GRAY2BGR), concat_img32]))

        cv2.waitKey(-1)

    def line_test2(self):
        # left_img_names = glob.glob('./statue_back_side/pattern_cam1_*.jpg')
        # right_img_names = glob.glob('./statue_back_side/pattern_cam2_*.jpg')

        left_img_names = glob.glob('./captured/left-*.png')
        right_img_names = glob.glob('./captured/right-*.png')


        # left_img_names = glob.glob('./gray-row-col/pattern-*.png')
        # right_img_names = glob.glob('./gray-row-col/pattern-*.png')

        left_imgs=[]
        right_imgs = []
        white_imgs = []
        black_imgs = []
        target_imgs = []
        edge_imgs = []
        for index in range(0, len(left_img_names)):
            # cv2.remap
            if index < len(left_img_names) - 2:
                left_imgs.append(cv2.imread(left_img_names[index], cv2.IMREAD_GRAYSCALE))
                right_imgs.append(cv2.imread(right_img_names[index], cv2.IMREAD_GRAYSCALE))
            elif index == len(left_img_names) - 2:
                white_imgs.append(cv2.imread(left_img_names[index], cv2.IMREAD_GRAYSCALE))
                white_imgs.append(cv2.imread(right_img_names[index], cv2.IMREAD_GRAYSCALE))
            elif index == len(left_img_names) - 1:
                black_imgs.append(cv2.imread(left_img_names[index], cv2.IMREAD_GRAYSCALE))
                black_imgs.append(cv2.imread(right_img_names[index], cv2.IMREAD_GRAYSCALE))

        threshold_val = 120
        # _, white_imgs[0] = cv2.threshold(white_imgs[0], threshold_val, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        diff_val = 10
        black_tol = 20
        left_shadow_mask, right_shadow_mask = self.compute_shadow_mask(white_imgs, black_imgs)
        cv2.imwrite('left_shadow_mask.png', left_shadow_mask)
        cv2.imwrite('right_shadow_mask.png', right_shadow_mask)
        # norm_left_shadow_mask = cv2.normalize(left_shadow_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # norm_right_shadow_mask = cv2.normalize(right_shadow_mask, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        dec_img1, dec_img2 = self.get_projection_pixel(left_imgs, left_shadow_mask)
        print('berry_test')



    def compute_shadow_mask(self, whites, blacks):
        shadow_maks_list = []
        for index in range(0, len(whites)):
            shadow_mask = np.where(abs(whites[index] - blacks[index])>self.black_threshold, 255, 0)
            shadow_maks_list.append(shadow_mask.astype(np.uint8))

        return shadow_maks_list

    def get_projection_pixel(self, pattern_imgs, shadow_mask):
        original_imgs = copy.deepcopy(pattern_imgs)
        gray_img_list = []
        gray_for_viz_list = []
        gray_for_viz_color_list = []

        col_imgs = math.ceil(math.log(self.PATTERN_WIDTH) / math.log(2))
        row_imgs = math.ceil(math.log(self.PATTERN_HEIGHT) / math.log(2))

        for index in range(0, col_imgs+row_imgs):
            even_img = original_imgs[index*2]
            odd_img = original_imgs[index * 2+1]
            # 원본값
            # img = np.where((shadow_mask ==255) & (even_img > odd_img), 1, 0)
            # #-1=error
            # img = np.where((shadow_mask ==255) & (abs(even_img-odd_img)< self.white_threshold), -1, img)
            # #-2=shadow인 곳
            # img = np.where(shadow_mask == 0, -2, img)

            img = np.where((shadow_mask ==255) & (even_img > odd_img), 255, 0)
            img = np.where((shadow_mask ==255) & (abs(even_img-odd_img)< self.white_threshold), 0, img)
            img = np.where(shadow_mask == 0, 0, img)

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
        dec_img1 = self.gray_to_deciaml_num(gray_img_list[:col_imgs])
        dec_img2 = self.gray_to_deciaml_num(gray_img_list[col_imgs:])
        return dec_img1, dec_img2

    def gray_to_deciaml_num(self, gray_img_list):
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





if __name__ == '__main__':

    main = main()
    mode = 0
    # # main.line_test2()
    #
    # c1 = Camera(serial=13142459)
    # c1.connect()
    #
    # c1.start_capture()
    # print('c1: ', c1.is_color)
    #
    # while True:
    #     cv2.waitKey(500)
    #     c1.read_next_image()
    #
    #     image1 = c1.get_current_image()  # last image
    #     imageData1 = np.asarray(image1["buffer"], dtype=np.byte)
    #     rows = image1["rows"]
    #     cols = image1["cols"]
    #     cv_image1 = np.array(image1["buffer"], dtype="uint8").reshape((image1["rows"], image1["cols"]))
    #
    #     rowBytes = image1["received_size"] / image1["rows"]
    #     test_img = cv2.UMat(rows, cols, cv2.CV_8UC3, np.array(image1["buffer"], dtype="uint8"),rowBytes )
    #
    #     print(rows)
    #     # cvImage = cv::Mat(cf2Img.GetRows(), cf2Img.GetCols(), CV_8UC3, cf2Img.GetData(), rowBytes);
    #


    while True:
        mode = 0
        if main.thread_run2==False:
            mode = input("1: 캡쳐 모드\n2: 캘리브레이션 모드\n3: 3차원 복원 모드\n4: 스트리밍 모드\n5: 패턴생성\n6: 테스트\n7: 종료\n\n모드: ")
        else:
            mode = input("1: 캡쳐 모드\n2: 캘리브레이션 모드\n3: 3차원 복원 모드\n4: 스트리밍 종료\n5: 패턴생성\n6: 테스트\n7: 종료\n\n모드: ")
        if mode == '1':
            ret = main.cam_capture()
            if ret is False:
                continue
        elif mode == '2':

            pass
        elif mode == '3':
            dataset_name = input('데이터셋 종류: ')
            main.structured_light_reconstruction(dataset_name)

        elif mode == '4':
            if main.thread_run2 == False:
                main.cam_stream()
            else:
                main.thread_run2 = False

        elif mode == '5':
            main.generate_pattern()

        elif mode == '6':
            main.line_test2()

        elif mode == '7':
            print('종료')
            break
        else:
            print('잘못된 입력입니다')
            continue


