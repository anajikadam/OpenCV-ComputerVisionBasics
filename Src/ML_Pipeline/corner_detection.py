import numpy as np
import cv2 as cv


class CornerDetection:

    def __init__(self, path):
        self.image_path = path

    def detect(self):

        """
        For arguments for cornerHarris:
          img - Input image. It should be grayscale and float32 type.
          blockSize - It is the size of neighbourhood considered for corner detection
          ksize - Aperture parameter of the Sobel derivative used.
          k - Harris detector free parameter in the equation.
        """

        img = cv.imread(self.image_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        dst = cv.cornerHarris(gray, 2, 3, 0.04)
        # result is dilated for marking the corners, not important
        dst = cv.dilate(dst, None)
        # Threshold for an optimal value, it may vary depending on the image.
        img[dst > 0.01 * dst.max()] = [0, 0, 255]
        cv.imshow('dst', img)
        if cv.waitKey(0) & 0xff == 27:
            cv.destroyAllWindows()
