import numpy as np
import cv2
def generate(nx, ny, radius, offset, margin_x = 20, margin_y = 20, resolution_x=1280, resolution_y=800):
    black = np.zeros((resolution_y, resolution_x))
    # circle_img = None
    for x in range(0, nx):
        for y in range(0, ny):
            center_x = margin_x+x*offset
            center_y = margin_y+y*offset
            center_x = np.clip(center_x, 0, resolution_x)
            center_y = np.clip(center_y, 0, resolution_y)

            black = cv2.circle(black, (center_x, center_y), radius, (255,255,255), -1, cv2.LINE_AA)

    cv2.namedWindow("Circle Window", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Circle Window", 1920, 0)
    cv2.setWindowProperty("Circle Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('Circle Window', black)
    cv2.imwrite('circle_pattern.png', black)
    cv2.waitKey(-1)
    cv2.destroyWindow('Circle Window')

