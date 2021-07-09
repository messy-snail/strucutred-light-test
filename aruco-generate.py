from cv2 import aruco
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
cb = aruco.CharucoBoard_create(10, 14, 20, 14, aruco_dict)
print(cb.dictionary)
workdir = "data/"


imboard = board.draw((500, 500))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.imshow(imboard, cmap = mpl.cm.gray, interpolation = "nearest")
ax.axis("off")
# plt.savefig(workdir + "chessboard.pdf")
cv2.imshow("test", imboard)
cv2.imwrite("chessboard.png", imboard)
plt.show()