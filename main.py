from pyflycap2.interface import GUI
from pyflycap2.interface import Camera

from _thread import*

import numpy as np
import cv2
import glob

class main:
    def __init__(self):
        self.cam_serial1 = 0
        self.cam_serial2 = 0
        self.capture_name = ''

        # self.PATTERN_WIDTH = 1280
        # self.PATTERN_HEIGHT = 800

        self.PATTERN_WIDTH = 160
        self.PATTERN_HEIGHT = 60
        self.CAPTURE_FLAG = False

        self.CAPTURE_INDEX = 0

        self.thread_run = False
        self.thread_run2 = False

    def cam_thread(self, id):
        self.capture_name1 = 'left'
        self.capture_name2 = 'right'

        c1 = Camera(serial=self.cam_serial1)
        c2 = Camera(serial=self.cam_serial2)

        c1.connect()
        c1.start_capture()

        c2.connect()
        c2.start_capture()

        while self.thread_run == True:
            cv2.waitKey(500)
            c1.read_next_image()
            image1 = c1.get_current_image()  # last image
            imageData1 = np.asarray(image1["buffer"], dtype=np.byte)
            cv_image1 = np.array(image1["buffer"], dtype="uint8").reshape((image1["rows"], image1["cols"]));

            c2.read_next_image()
            image2 = c2.get_current_image()  # last image
            imageData2 = np.asarray(image2["buffer"], dtype=np.byte)
            cv_image2 = np.array(image2["buffer"], dtype="uint8").reshape((image1["rows"], image1["cols"]));

            cv2.imshow(f'left-{id}', cv_image1)
            cv2.imshow(f'right-{id}', cv_image2)
            cv2.waitKey(10)

            if self.CAPTURE_FLAG:
                ret1 = False
                ret1 = cv2.imwrite(f'./captured/{self.capture_name1}-{self.CAPTURE_INDEX:02d}.png', cv_image1)
                ret2 = False
                ret2 = cv2.imwrite(f'./captured/{self.capture_name2}-{self.CAPTURE_INDEX:02d}.png', cv_image2)

                while True:
                    if ret1==True and ret2==True:
                        self.CAPTURE_FLAG = False
                        self.CAPTURE_INDEX+=1
                        break
        c1.disconnect()
        c2.disconnect()
        self.CAPTURE_INDEX = 0
    def cam_thread2(self, id):
        self.capture_name1 = 'left'
        self.capture_name2 = 'right'

        c1 = Camera(serial=self.cam_serial1)
        c2 = Camera(serial=self.cam_serial2)

        c1.connect()
        c1.start_capture()

        c2.connect()
        c2.start_capture()

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

        self.cam_serial1 = 13142459
        self.cam_serial2 = 14193278
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


    def generate_pattern(self):
        gray_code_generator = cv2.structured_light.GrayCodePattern_create(self.PATTERN_WIDTH, self.PATTERN_HEIGHT)
        _, patter_images = gray_code_generator.generate()
        patter_images.append(255 * np.ones((self.PATTERN_HEIGHT, self.PATTERN_WIDTH)))
        patter_images.append(np.zeros((self.PATTERN_HEIGHT, self.PATTERN_WIDTH)))
        print(f'Pattern Size: {len(patter_images)}')
        num = 0
        for image in patter_images:
            cv2.imwrite(f"pattern/pattern-{num}.png", image)
            num += 1



if __name__ == '__main__':

    main = main()
    mode = 0

    while True:
        mode = 0
        if main.thread_run2==False:
            mode = input("1: 캡쳐 모드\n2: 캘리브레이션 모드\n3: 3차원 복원 모드\n4: 스트리밍 모드\n5: 패턴생성\n6: 종료\n\n모드: ")
        else:
            mode = input("1: 캡쳐 모드\n2: 캘리브레이션 모드\n3: 3차원 복원 모드\n4: 스트리밍 종료\n5: 패턴생성\n6: 종료\n\n모드: ")
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
            break
        elif mode == '6':
            print('종료합니다')
            break
        else:
            print('잘못된 입력입니다')
            continue


