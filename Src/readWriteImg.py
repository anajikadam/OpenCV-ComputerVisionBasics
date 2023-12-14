from ML_Pipeline.airthmetic_operations import *
from ML_Pipeline.colorspaces import *
from ML_Pipeline.corner_detection import *
from ML_Pipeline.edge_detection import *
from ML_Pipeline.face_mouth_detection import *
from ML_Pipeline.hough_line_circle_detection import *
from ML_Pipeline.matcher import *
from ML_Pipeline.morphology import *
from ML_Pipeline.read_write import *
from ML_Pipeline.admin import input_folder
from ML_Pipeline.SIFT_FeatureTransform import *
from ML_Pipeline.smoothing import *
from ML_Pipeline.template_matching import *
from ML_Pipeline.thresholding import *
from ML_Pipeline.video_processing import *
import os

# reading and writing images
image_path = os.path.join(input_folder, "./test.jpg")
read_write_obj = ReadWriteDisplay(image_path)
read_write_obj.show()
read_write_obj.read()
read_write_obj.write()


# python readWriteImg.py
# Shape of the image: (280, 390, 3)
# Shape of the grayscale image: (280, 390)
