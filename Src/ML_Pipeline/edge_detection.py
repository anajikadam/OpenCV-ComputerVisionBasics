import cv2
from .admin import output_folder
import os


class CannyEdgeDetection:
    def __init__(self, path):
        self.image_path = path

    def canny_edge(self):
        """Second and third arguments are our minVal and maxVal """
        """minval and maxval are used for Hysteresis Thresholding """

        img = cv2.imread(self.image_path, 0)
        edges = cv2.Canny(img, 50, 200)
        cv2.imwrite(os.path.join(output_folder, "edges.jpg"), edges)


