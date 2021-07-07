import structuredlight as sl
import os
import cv2
import numpy as np
import stripe_pattern
# width = 1024
# width = 512
width = 1280
height = 800


pattern_name = 'b'

if not os.path.isdir(pattern_name):
    os.mkdir(pattern_name)

if pattern_name == 'binary' or pattern_name == 'bin'or pattern_name == 'b':
    binary = sl.Binary()
    # imlist = binary.generate((width, height))
    imlist,imlist2  = stripe_pattern.generate((width, height))

    index = 0
    cv2.namedWindow("Pattern Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pattern Window", width, height)
    cv2.moveWindow("Pattern Window", 1920, 0)
    cv2.setWindowProperty("Pattern Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for index in range(0, len(imlist)):
        cv2.imshow("Pattern Window", imlist[index])
        cv2.waitKey(100)

    for index in range(0, len(imlist2)):
        cv2.imshow("Pattern Window", imlist2[index])
        cv2.waitKey(100)
        # cv2.imwrite(f'./{pattern_name}/{index:03d}-pattern.png', imlist[index])
        # index+=1

elif pattern_name == 'gray'or pattern_name == 'g':
    gray = sl.Gray()
    imlist = gray.generate((width, height))
    imlist.append(255 * np.ones((height, width)))
    imlist.append(np.zeros((height, width)))

    index = 0
    cv2.namedWindow("Pattern Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pattern Window", width, height)
    cv2.moveWindow("Pattern Window", 1920, 0)
    cv2.setWindowProperty("Pattern Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for img in imlist:
        cv2.imshow("Pattern Window", img)
        cv2.waitKey(10)
        cv2.imwrite(f'./{pattern_name}/{index:03d}-pattern.png', img)
        index += 1

    img_index = gray.decode(imlist, thresh=127)
elif pattern_name == 'stripe'or pattern_name == 's':
    stripe = sl.Stripe()
    imlist = stripe.generate((width, height))

    index = 0
    cv2.namedWindow("Pattern Window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pattern Window", width, height)
    cv2.moveWindow("Pattern Window", 1920, 0)
    cv2.setWindowProperty("Pattern Window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    for img in imlist:
        cv2.imshow("Pattern Window", img)
        cv2.waitKey(20)
        cv2.imwrite(f'./{pattern_name}/{index:03d}-pattern.png', img)
        index += 1

    img_index = stripe.decode(imlist)

print(img_index)