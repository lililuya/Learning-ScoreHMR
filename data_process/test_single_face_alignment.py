import face_alignment
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)


image = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n000132/0048_01.jpg"

image1 = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n000002/0018_04.jpg"

image2 = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n000004/0294_01.jpg"

image3 = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n004080/0344_01.jpg"

image = cv2.imread(image3)
lmks, _, bbox = face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True) # 返回三项lmks、score、face bbox
# 对多人脸和None要有一个判断

if bbox is None:
    lmks = None
else:
    lmks = lmks[0]
print(bbox is None)
# 左上角的 x 坐标、左上角的 y 坐标、右下角的 x 坐标、右下角的 y 坐标
# crop_bbox = get_bbox(image, lmks)
# x_la, y_la, x_rb, y_rb = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
# # print(x_la, x_rb, y_la, y_rb)
# # crop_image  = crop_image_bbox(image, crop_bbox)
# crop_image = image[max(y_la, 0): y_rb, max(x_la, 0): x_rb, :]
# # print(crop_image)

# cv2.imwrite("/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_detect/test_single_crop_cornerl.png", crop_image)
# print("finish crop image")
