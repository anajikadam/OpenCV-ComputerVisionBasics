import cv2
import numpy as np
from .admin import output_folder
import os


def check_color_spaces():
    flags = [i for i in dir(cv2) if i.startswith('COLOR_')]
    print(flags)


class ColorSpaces:
    """mainly we tend to convert from BGR->Gray and BGR-> HSV"""
    """if you want to see all of the color spaces, please run check_color_spaces"""

    def __init__(self, path):
        self.image_path = path
        self.image = cv2.imread(self.image_path)

    def convert_bgr_hsv(self):
        """converts bgr image to hsv"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        cv2.imwrite(os.path.join(output_folder, "bgr2hsv.jpg"), hsv)

    def convert_bgr_gray(self):
        """converts bgr image to gray"""

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(output_folder, "bgr2gray.jpg"), gray)

    def track_blue(self):
        """tracks blue color,first convert into hsv and then we go bitwise and to detect the blue"""
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        lower_blue = np.array([25, 50, 50])
        upper_blue = np.array([130, 255, 255])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(self.image, self.image, mask=mask)
        cv2.imwrite(os.path.join(output_folder, "detected_blue.jpg"), res)
