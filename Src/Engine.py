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

# airthmetic operations on images
image_path_src = os.path.join(input_folder, "test.jpg")
image_path_dest = os.path.join(input_folder, "test02.jpg")
airthmetic_obj = AirthmeticOperations(image_path_src, image_path_dest)
airthmetic_obj.add()
airthmetic_obj.substract()

# # color spaces and changing color spaces
# # check all color spaces
# check_color_spaces()
image_path_src = os.path.join(input_folder, "blue_cap.jpg")
color_spaces_obj = ColorSpaces(image_path_src)
color_spaces_obj.convert_bgr_gray()
color_spaces_obj.convert_bgr_hsv()
color_spaces_obj.track_blue()

# image thresholding
image_path_src = os.path.join(input_folder, "blue_cap.jpg")
thres_obj = Thresholding(image_path_src)
thres_obj.simple_thres()
thres_obj.adaptive_thres()
thres_obj.otsu_binarization()

# smoothing images
image_path_src = os.path.join(input_folder, "test.jpg")
smoothing_object = Smoothing(image_path_src)
smoothing_object.averaging_numpy()
smoothing_object.median_blur()
smoothing_object.gaussian_blur()
smoothing_object.bilateral_blur()
smoothing_object.average_bluring()

# Morphological transformation
image_path_src = os.path.join(input_folder, "test.jpg")
morpho_obj = Morphological(image_path_src)
morpho_obj.erode()
morpho_obj.dilate()
morpho_obj.opening()
morpho_obj.closing()
morpho_obj.get_structuring_element()

# canny edge detection
image_path_src = os.path.join(input_folder, "test.jpg")
canny_edge_obj = CannyEdgeDetection(image_path_src)
canny_edge_obj.canny_edge()

# #template matching
image_path_src = os.path.join(input_folder, "test.jpg")
template_obj = TemplateMatching(image_path_src,image_path_src)
template_obj.template_matching()

# multi scale template matching
template_obj.multiscale_template_matching()

# Hough Transforms for Line and circle
image_path_src = os.path.join(input_folder, "test.jpg")
hough_object = HoughLineCircleDetection(image_path_src)
hough_object.line_detection()
hough_object.circle_detection()

# video processing
video_path = os.path.join(input_folder, "test_video.mp4")
video_obj = VideoAnalytics(video_path)
video_obj.process()

# # harris corner detection
image_path_src = os.path.join(input_folder, "test.jpg")
corner_obj = CornerDetection(image_path_src)
corner_obj.detect()

# # sift feature detection
image_path_src = os.path.join(input_folder, "test.jpg")
sift_obj = SIFT(image_path_src)
sift_obj.drawKeypoints()
sift_obj.match(image_path_src, image_path_src)

# # feature matching using flann and brute-force orb
image_path_src = os.path.join(input_folder, "test.jpg")
image_path_dest = os.path.join(input_folder, "test.jpg")
matcher_obj = Matcher(image_path_src, image_path_dest)
matcher_obj.brute_force_matcher()
matcher_obj.flann_matcher()

# face-eye detection
face_detection_obj = detect()
