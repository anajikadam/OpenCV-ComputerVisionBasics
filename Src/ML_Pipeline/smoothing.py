import cv2
import numpy as np
from .admin import output_folder
import os


class Smoothing:
    def __init__(self, path):
        self.image_path = path
        self.image = cv2.imread(self.image_path)

    def averaging_numpy(self):
        """this is the numpy implementation of average blur using open cv"""
        kernel = np.ones((5, 5), np.float32) / 25
        dst = cv2.filter2D(self.image, -1, kernel)

        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "averaged_numpy.jpg"), dst)

    def average_bluring(self):
        blur = cv2.blur(self.image, (5, 5))
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "averaged_blur.jpg"), blur)

    def gaussian_blur(self):
        """instead of box kernel, gaussian kernel is used"""
        """ width and height of the kernel which should be positive and odd. """
        """sigmaX and sigmaY respectively. If only sigmaX is specified, sigmaY is taken as the same as sigmaX. If 
        both are given as zeros, they are calculated from the kernel size """

        blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "gaussian_blur.jpg"), blur)

    def median_blur(self):
        """ median of all the pixels under the kernel area and the central element is replaced with this median value"""
        """highly effective in salt and pepper noise"""
        """Kernel size should be odd number"""

        median = cv2.medianBlur(self.image, 5)
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "median_blur.jpg"), median)

    def bilateral_blur(self):
        """highly effective in noise removal while keeping edges sharp"""
        """Gaussian blurs the edges also,bilateral doesn't ,this is why it is kind of slow to gaussian"""
        """
        d: diameter of each pixel neighborhood that is used during filtering 
        sigmaColor: Filter sigma in the color 
            space. A larger value of the parameter means that farther colors within the pixel neighborhood .
        sigmaSpace:Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels 
            will influence each other as long as their colors are close enough """

        blur = cv2.bilateralFilter(self.image, 9, 75, 75)
        cv2.imwrite(os.path.join(output_folder, "original.jpg"), self.image)
        cv2.imwrite(os.path.join(output_folder, "bilateral_blur.jpg"), blur)

