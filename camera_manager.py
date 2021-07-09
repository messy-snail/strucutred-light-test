from pyflycap2.interface import Camera

from _thread import*

import numpy as np
import cv2
import aruco_test
# import pattern_manager
import time

import common as com
class camera_manager:
    def __init__(self):
        self.CAPTURE_INDEX = 0
        self.CAPTURE_FLAG = False
        self.ARUCO_FLAG = False

        # self.CAM_SERIAL = 13142459
        self.CAM_SERIAL = 14193278
        self.THREAD_RUN = False
        self.EXPOSURE_TIME = 50

        self.camera_instance = None

        # self.pm = None

        self.cv_image1 = None
        self.capture_name = 'left'

        self.LBAEL_ORIGINAL = None
        self.LABEL_PATTERN = None
        self.LOG = None
    def __del__(self):
        if self.camera_instance is not None:
            self.camera_instance.disconnect()
        pass

    def get_image(self):
        return self.cv_image1

    def cam_thread(self, id):
        while self.THREAD_RUN == True:
            if self.CAPTURE_FLAG:
                com.pm.start()
                # self.pm.CAPTURE_FLAG = True
                for index in range(0, len(com.pm.patter_images)):
                    com.pm.next_pattern(index)
                    cv2.waitKey(450)
                    self.camera_instance.read_next_image()
                    image1 = self.camera_instance.get_current_image()  # last image
                    image_data1 = np.asarray(image1["buffer"], dtype=np.uint8)
                    bayer_img = image_data1.reshape((image1["rows"], image1["cols"]));
                    self.cv_image1 = cv2.cvtColor(bayer_img, cv2.COLOR_BayerGB2RGB)
                    cv2.imwrite(f'./captured/{self.capture_name}-{index:02d}.png', self.cv_image1)
                    com.lm.view_original_image(com.LABEL_ORIGINAL, self.cv_image1, True)
                    com.log.print_log(f'{index:02d}번째 사진이 캡쳐되었습니다', com.TE_LOG)
                com.pm.destroy_pattern_window()
                    # self.pm.CAPTURE_DONE = True
                self.CAPTURE_FLAG = False

            cv2.waitKey(50)
            self.camera_instance.read_next_image()
            image1 = self.camera_instance.get_current_image()  # last image
            image_data1 = np.asarray(image1["buffer"], dtype=np.uint8)
            bayer_img = image_data1.reshape((image1["rows"], image1["cols"]));
            self.cv_image1 = cv2.cvtColor(bayer_img, cv2.COLOR_BayerGB2RGB)
            aruco_img = None
            if self.ARUCO_FLAG:
                aruco_img = aruco_test.detect_aruco(self.cv_image1)

            if aruco_img is not None:
                com.lm.view_original_image(com.LABEL_TARGET, aruco_img, True)
            com.lm.view_original_image(com.LABEL_ORIGINAL, self.cv_image1, True)

        self.camera_instance.disconnect()
        self.CAPTURE_INDEX = 0

    def cam_open(self):
        try:
            self.camera_instance = Camera(serial=self.CAM_SERIAL)
            self.camera_instance.connect()
            self.camera_instance.start_capture()
            self.camera_instance.set_cam_setting_option_values('exposure',self.EXPOSURE_TIME)
            print('color: ', self.camera_instance.is_color)
            self.THREAD_RUN = True
            ret = start_new_thread(self.cam_thread, (0,))
        except Exception as e:
            self.THREAD_RUN = False
            return False


        return True
    def cam_close(self):
        self.thread_run = False

    def cam_auto_capture(self):
        self.CAPTURE_FLAG = True

        self.thread_run = False
        return True

    def cam_manual_capture(self):
        self.CAPTURE_FLAG = True

        self.thread_run = False
        return True