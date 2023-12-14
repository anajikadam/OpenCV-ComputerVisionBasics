import cv2
import sys
import os
from .admin import output_folder


class ReadWriteDisplay:
    def __init__(self, image_path):
        self.image_path = image_path

    def read(self, ):
        """ cv2.imread takes a image path to be read"""
        image = cv2.imread(self.image_path)
        shape = image.shape
        # shape goes by : (rows,cols,channels)
        print("Shape of the image:", shape)

        """ reading image with flags"""
        # cv2.IMREAD_COLOR (default) ,cv2.IMREAD_GRAYSCALE (0) and cv2.IMREAD_UNCHANGED(-1)

        grayscale = cv2.imread(self.image_path, 0)
        # shape goes by : (rows,cols,channels)
        shape = grayscale.shape
        print("Shape of the grayscale image:", shape)

    def write(self):
        image = cv2.imread(self.image_path)
        if image is None:
            sys.exit("Could not read the image.")
        cv2.imwrite(os.path.join(output_folder, "write_output.jpg"), image)

    def show(self):
        image = cv2.imread(self.image_path)
        cv2.imshow("Display", image)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
