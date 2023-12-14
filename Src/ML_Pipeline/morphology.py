import cv2
import numpy as np
from .admin import output_folder
import os


class Morphological:
    def __init__(self, path):
        self.image_path = path
        self.image = cv2.imread(self.image_path, 0)

    def erode(self):
        """erodes away the boundaries of foreground object (Always try to keep foreground in white)."""
        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(self.image, kernel, iterations=1)
        cv2.imwrite(os.path.join(output_folder, "erosed.jpg"), erosion)

    def dilate(self):
        """It increases the white region in the image or size of foreground object increases"""
        kernel = np.ones((5, 5), np.uint8)
        dilation = cv2.dilate(self.image, kernel, iterations=1)
        cv2.imwrite(os.path.join(output_folder, "dilated.jpg"), dilation)

    def opening(self):
        """Opening is just another name of erosion followed by dilation. It is useful in removing noise"""
        kernel = np.ones((5, 5), np.uint8)
        opening = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        cv2.imwrite(os.path.join(output_folder, "opened.jpg"), opening)

    def closing(self):
        """Closing is reverse of Opening, Dilation followed by Erosion. It is useful in closing small holes inside
        the foreground objects, or small black points on the object. """
        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, kernel)
        cv2.imwrite(os.path.join(output_folder, "closing.jpg"), closing)

    def get_structuring_element(self):
        """get structuring  element to work on above morphological operation if required"""
        rect_element = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        print(rect_element)
        elliptical_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        print(elliptical_element)
