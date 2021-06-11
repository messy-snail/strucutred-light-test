from pyflycap2.interface import GUI
from pyflycap2.interface import Camera

from _thread import*

import numpy as np
import cv2
import sys

cam_serial1 = 14193278
cam_serial2 = None

PATTERN_WIDTH = 1280
PATTERN_HEIGHT = 800
CAPTURE_FLAG = False
CAPTURE_INDEX = 0
# gui = GUI()
# gui.show_selection()

def cam_thread(id):
    global CAPTURE_FLAG
    global CAPTURE_INDEX
    if id==0:
        c = Camera(serial=cam_serial1)
    else:
        c = Camera(serial=cam_serial2)
    c.connect()
    c.start_capture()

    while True:
        c.read_next_image()
        image = c.get_current_image()  # last image
        imageData = np.asarray(image["buffer"], dtype=np.byte)
        cv_image = np.array(image["buffer"], dtype="uint8").reshape((image["rows"], image["cols"]));
        if CAPTURE_FLAG:
            ret = cv2.imwrite(f'cam1-{CAPTURE_INDEX}.png', cv_image)
            if ret:
                CAPTURE_INDEX += 1
                CAPTURE_FLAG = False

        cv2.imshow('cam1', cv_image)
        cv2.waitKey(10)

    c.disconnect()


if __name__ == '__main__':

    start_new_thread(cam_thread, (0,))

    gray_code_generator = cv2.structured_light.GrayCodePattern_create(PATTERN_WIDTH, PATTERN_HEIGHT)
    _, patter_images = gray_code_generator.generate()

    # black white 뿌림
    patter_images.append(np.zeros((PATTERN_HEIGHT, PATTERN_WIDTH)))
    patter_images.append(255*np.ones((PATTERN_HEIGHT, PATTERN_WIDTH)))
    cv2.namedWindow("Pattern Window", cv2.WINDOW_NORMAL);
    cv2.resizeWindow("Pattern Window", PATTERN_WIDTH, PATTERN_HEIGHT)
    cv2.moveWindow("Pattern Window", 1920, 0);
    cv2.setWindowProperty("Pattern Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);

    print(f'Pattern Size: {len(patter_images)}' )
    # while(True):
    #     pass
    num=0
    for image in patter_images:
        cv2.imwrite(f"pattern/pattern-{num}.png", image)
        num+=1


    for image in patter_images:
        cv2.imshow("Pattern Window", image)
        cv2.waitKey(10)
        CAPTURE_FLAG = True
        while CAPTURE_FLAG:
            pass


