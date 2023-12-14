import cv2
import os
from .admin import output_folder


class VideoAnalytics:
    def __init__(self, video_path):
        self.video_path = video_path

    def process(self):
        """
        Videocapture takes an video path to be read,
        if we give 0 as an argument, we can directly feed camera input
        """
        cam = cv2.VideoCapture(self.video_path)
        basepath = os.path.join(output_folder, 'data')
        try:
            if not os.path.exists(basepath):
                os.makedirs(basepath)
        except OSError:
            print('Error: Creating directory of data')

        currentframe = 0
        while True:
            # reading one frame at a time,and checkig if ret is True, if False,brak the loop as there are no frame now
            ret, frame = cam.read()
            if ret:
                name = os.path.join(basepath, 'frame' + str(currentframe) + '.jpg')
                print('Creating...' + name)
                cv2.imwrite(name, frame)
                currentframe += 1
            else:
                break

        cam.release()
        cv2.destroyAllWindows()
