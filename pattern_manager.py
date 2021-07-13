import cv2
import numpy as np
import os
import stripe_pattern

import common as com


class pattern_manager:
    def __init__(self):
        self.CAPTURE_MODE =False
        self.CAPTURE_FLAG = False

        self.patter_images = []
        self.LABEL_PATTERN = None
        self.PATTERN_INDEX = 0
        self.OPTION = 'g'


    def start(self):

        gray_code_generator = cv2.structured_light.GrayCodePattern_create(com.PATTERN_WIDTH, com.PATTERN_HEIGHT)
        # _, self.patter_images = gray_code_generator.generate()
        self.patter_images = stripe_pattern.generate((com.PATTERN_WIDTH, com.PATTERN_HEIGHT), self.OPTION)

        # _, self.patter_images = gray_code_generator.generate()
        # white black 뿌림
        self.patter_images.append(255 * np.ones((com.PATTERN_HEIGHT, com.PATTERN_WIDTH)))
        self.patter_images.append(np.zeros((com.PATTERN_HEIGHT, com.PATTERN_WIDTH)))
        cv2.namedWindow("Pattern Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Pattern Window", com.PATTERN_WIDTH, com.PATTERN_HEIGHT)
        # cv2.moveWindow("Pattern Window", 1920, 0)
        cv2.moveWindow("Pattern Window", -1920, 0)
        cv2.setWindowProperty("Pattern Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Pattern Window", self.patter_images[0])
        cv2.waitKey(300)
        print(f'Pattern Size: {len(self.patter_images)}')


    def next_pattern(self, index):
        cv2.imshow("Pattern Window", self.patter_images[index])
        cv2.imwrite(f'./captured/right-{index:02d}.png', self.patter_images[index])
        cv2.waitKey(100)
        return self.patter_images[index]

    def destroy_pattern_window(self):
        cv2.destroyWindow("Pattern Window")

    def label_thread(self, id):
        index = 0
        for image in self.patter_images:
            cv2.imshow("Pattern Window", image)

            pattern_name=''
            if self.OPTION=='g':
                pattern_name = 'gray'
            elif self.OPTION=='b':
                pattern_name = 'binary'

            if not os.path.exists(pattern_name):
                os.mkdir(pattern_name)

            cv2.imwrite(f'{pattern_name}/{index:02d}.png', image)
            com.log.print_log(f'{pattern_name}, {index:02d}번째 패턴이 저장되었습니다', com.TE_LOG)
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            com.lm.view_original_image(com.LABEL_TARGET, image, True)
            cv2.waitKey(100)
            index+=1
            self.PATTERN_INDEX = index

        self.destroy_pattern_window()




