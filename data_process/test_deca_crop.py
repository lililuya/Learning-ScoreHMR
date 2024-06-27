import face_alignment
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
from crop_tools_deca import crop
from skimage.transform import estimate_transform, warp

"""相关的配置信息"""

device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)


image = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n000002/0001_01.jpg"
image = cv2.imread(image)
lmks, _, bbox = face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True) # 返回三项lmks、score、face bbox
# 对多人脸和None要有一个判断
if bbox is None:
    lmks = None
else:
    lmks = lmks[0]


x_la, y_la, x_rb, y_rb = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])

# image = image/255.
kpt = lmks[:,:2]

tform = crop(image, lmks)
cropped_image = warp(image, tform.inverse, output_shape=(224, 224))
cropped_image = cropped_image*255

cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

# normalized kpt
cropped_kpt[:,:2] = cropped_kpt[:,:2]/224 * 2  - 1
cv2.imwrite("/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_detect/test_single_crop_deca.png", cropped_image)
print("finish crop image")
