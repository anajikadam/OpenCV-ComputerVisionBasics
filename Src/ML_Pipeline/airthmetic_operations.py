from .admin import output_folder
import os
import cv2


class AirthmeticOperations:
    def __init__(self, path01, path02):
        self.image_path_src = path01
        self.image_path_dest = path02

    def add(self):
        """
            cv2.addWeighted(img1, wt1, img2, wt2, gammaValue)
            gammaValue : measurement of light
        """
        image_src = cv2.imread(self.image_path_src)
        image_dest = cv2.imread(self.image_path_dest)

        # resizing both of images to same dimension before adding
        # take (1050,1610) as a size we need to resize
        image_src = cv2.resize(image_src, (1050, 1610))
        image_dest = cv2.resize(image_dest, (1050, 1610))

        weightedSum = cv2.addWeighted(image_src, 0.5, image_dest, 0.4, 0)
        cv2.imwrite(os.path.join(output_folder, "added.jpg"), weightedSum)

    def substract(self):
        """
        substract pixel wise one image from another image
        """
        image_src = cv2.imread(self.image_path_src)
        image_dest = cv2.imread(self.image_path_dest)

        # resizing both of images to same dimension before subtracting
        # take (1050,1610) as a size we need to resize
        image_src = cv2.resize(image_src, (1050, 1610))
        image_dest = cv2.resize(image_dest, (1050, 1610))

        remaining_image = cv2.subtract(image_src, image_dest)
        cv2.imwrite(os.path.join(output_folder, "substracted.jpg"), remaining_image)
