import cv2
from .admin import output_folder
import os


class Thresholding:
    def __init__(self, path):
        self.image_path = path
        self.image = cv2.imread(self.image_path, 0)

    def simple_thres(self):
        """If the pixel value is smaller than the threshold, it is set to 0, otherwise it is set to a maximum value."""
        """ can be used directly as cv2.threshold """
        """options can be 
            cv2.THRESH_BINARY
            cv2.THRESH_BINARY_INV
            cv2.THRESH_TRUNC
            cv2.THRESH_TOZERO
            cv2.THRESH_TOZERO_INV
        """
        ret, thresh1 = cv2.threshold(self.image, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(output_folder, "simple_thres.jpg"), thresh1)

    def adaptive_thres(self):
        """Should be used where  an image has different lighting conditions in different areas.algorithm determines
        the threshold for a pixel based on a small region around it """
        """Threshold can be calculated as given two methods:
         1) cv2.ADAPTIVE_THRESH_MEAN_C: The threshold value is 
                the mean of the neighbourhood area minus the constant C.
         2) cv2.ADAPTIVE_THRESH_GAUSSIAN_C:The threshold 
                value is a gaussian-weighted sum of the neighbourhood values minus the constant C.  
        """
        img = cv2.medianBlur(self.image, 5)
        th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(os.path.join(output_folder, "adaptive_thres.jpg"), th2)

    def otsu_binarization(self):
        """Consider an image with only two distinct image values (bimodal image), where the histogram would only
        consist of two peaks. A good threshold would be in the middle of those two values. Similarly, Otsu's method
        determines an optimal global threshold value from the image histogram """

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(self.image, (5, 5), 0)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imwrite(os.path.join(output_folder, "otsu_binarized_thres.jpg"), th3)
