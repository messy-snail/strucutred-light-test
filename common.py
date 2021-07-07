import glob
import cv2

import log_manager
import label_manager
import pattern_manager


PATTERN_WIDTH = 640
PATTERN_HEIGHT = 480

log = log_manager.log_manager()
lm = label_manager.label_manager()
pm = pattern_manager.pattern_manager()

LABEL_ORIGINAL = None
LABEL_TARGET = None

TE_LOG = None

def read_captured_imgs():
    left_imgs = []
    right_imgs = []
    white_imgs = []
    black_imgs = []

    left_img_names = glob.glob('./captured/left-*.png')
    right_img_names = glob.glob('./captured/right-*.png')

    # left_img_names = glob.glob('./captured/pattern_cam1_*.jpg')
    # right_img_names = glob.glob('./captured/pattern_cam2_*.jpg')

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

    return left_imgs, right_imgs, white_imgs, black_imgs

def read_extracted_imgs():
    imgs = []
    img_names = glob.glob('./gray_img/*.png')
    for index in range(0, len(img_names)):
        imgs.append(cv2.imread(img_names[index]))
    return imgs

def read_chessboard_imgs():
    imgs = []
    img_names = glob.glob('./chessboard/*.png')
    for index in range(0, len(img_names)):
        imgs.append(cv2.imread(img_names[index]))
    return imgs

# PATTERN_WIDTH = 1920
# PATTERN_HEIGHT = 1080