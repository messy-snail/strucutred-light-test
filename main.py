from pyflycap2.interface import GUI
from pyflycap2.interface import Camera

from _thread import*

import numpy as np
import cv2
import glob

class main:
    def __init__(self):
        self.cam_serial = 0
        self.capture_name = ''

        self.PATTERN_WIDTH = 1280
        self.PATTERN_HEIGHT = 800
        self.CAPTURE_FLAG = False

        self.CAPTURE_INDEX = 0

        self.thread_run=False

    def cam_thread(self, id):
        print(self.cam_serial)
        self.capture_name=''


        if self.cam_serial == 13142459:
            self.capture_name = 'left'
        else:
            self.capture_name = 'right'

        if id==0:
            c = Camera(serial=self.cam_serial)

        c.connect()
        c.start_capture()

        while self.thread_run == True:
            c.read_next_image()
            image = c.get_current_image()  # last image
            imageData = np.asarray(image["buffer"], dtype=np.byte)
            cv_image = np.array(image["buffer"], dtype="uint8").reshape((image["rows"], image["cols"]));

            cv2.imshow(f'cam-{id}', cv_image)
            cv2.waitKey(10)

            if id == 0 and self.CAPTURE_FLAG:
                ret1 = False
                ret1 = cv2.imwrite(f'./captured/{self.capture_name}-{self.CAPTURE_INDEX:02d}.png', cv_image)
                cv2.waitKey(100)
                if ret1:
                    self.CAPTURE_FLAG = False
                    self.CAPTURE_INDEX+=1

            cv2.imshow(f'cam-{id}', cv_image)
            cv2.waitKey(10)

        c.disconnect()
        self.CAPTURE_INDEX = 0

    def cam_capture(self):

        selector = input('카메라 ID: ')
        if selector == 'l' or selector =='left' or selector =='13142459':
            self.cam_serial = 13142459
        elif selector == 'r' or selector =='right' or selector =='14193278':
            self.cam_serial = 14193278
        else:
            print('잘못된 입력입니다')
            return False
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
            cv2.waitKey(200)
            self.CAPTURE_FLAG = True
            # CAPTURE_FLAG2 = True
            while self.CAPTURE_FLAG:
                continue

        self.thread_run = False
        return True

    def structured_light_reconstruction(self):
        left_img_list = glob.glob('./captured/left-*.png')
        right_img_list = glob.glob('./captured/right-*.png')

        gray_code_generator = cv2.structured_light.GrayCodePattern_create(self.PATTERN_WIDTH, self.PATTERN_HEIGHT)

        gray_code_generator.setWhiteThreshold(200)
        gray_code_generator.setBlackThreshold(50)

        print(left_img_list)
        print(right_img_list)






if __name__ == '__main__':

    main = main()
    while True:
        mode = input("1: 캡쳐 모드\n2: 캘리브레이션 모드\n3: 3차원 복원 모드\n4: 종료\n\n모드: ")
        if mode == '1':
            ret = main.cam_capture()
            if ret is False:
                continue


        elif mode == '2':

            pass
        elif mode == '3':
            main.structured_light_reconstruction()
            pass
        elif mode == '4':
            print('종료합니다')
            break
        else:
            print('잘못된 입력입니다')
            continue



