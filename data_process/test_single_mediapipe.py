
import torch.nn.functional as F
from face_detector import FaceDetector
import logging
import cv2

"""目前有些在mediapipe中检测的dense——lmk不太准确"""

face_detector_mediapipe = FaceDetector('google')
logging.info("Mediapipe detector load successful!")
image_path = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n007082/0289_01.jpg"
image_path = 'liwen/DATA_for_HMR/VggFace2/source_data_train/n007082/0291_01.jpg'
image = cv2.imread(image_path)

dense_lmk = face_detector_mediapipe.dense(image)
print(dense_lmk is None)