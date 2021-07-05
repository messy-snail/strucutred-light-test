from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap, QImage
import copy
import cv2

class label_manager():
    def __init__(self):
        self.default_path = './_debug'

    def view_original_image(self, label, mat, swap):
        h=0
        w=0
        c=0

        if len(mat.shape) == 2:
            mat = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)

        h, w, c = mat.shape

        image = copy.deepcopy(mat)
        if swap==True:
            img  = QImage(image.data, w, h, w*c, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img.rgbSwapped()))

        else:
            img = QImage(image.data, w, h, w*c, QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        label.show()