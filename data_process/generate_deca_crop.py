import face_alignment
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F
from crop_tools_deca import crop
from skimage.transform import estimate_transform, warp
from face_detector import FaceDetector
import logging

# 设置日志等级
logging.basicConfig(level=logging.INFO)

device = "cuda" if torch.cuda.is_available() else "cpu"
face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
logging.info("FAN detector load successful!")
face_detector_mediapipe = FaceDetector('google')
logging.info("Mediapipe detector load successful!")

backlist_mp_path = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/backlist/backlist_mediapipe.txt"
backlist_68_path = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/backlist/backlist_68.txt"

def generate_crop_image(image_dir, lmk_dir, dense_lmk_dir, tform_dir, out_par_dir="/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_generate_deca_crop"):
    for image in os.listdir(image_dir):
        image_basename = os.path.splitext(image)[0]
        image_path = os.path.join(image_dir, image)
        real_path = os.path.relpath(os.path.dirname(image_path), image_dir) # 子目录
        image = cv2.imread(image_path)
        lmks, _, bbox = face_detector.get_landmarks_from_image(image_path, return_landmark_score=True, return_bboxes=True) # 返回三项lmks、score、face bbox
        dense_lmk = face_detector_mediapipe.dense(image)
        if dense_lmk is None:
            with open(backlist_mp_path, "w+") as mp_file:
                mp_file.write(image_path + "\n")
                print(image_path, " is detected none mp landmark")
                continue

        if bbox is None:
            with open(backlist_68_path, "w+") as ld_file:
                ld_file.write(image_path + "\n")
                print(image_path, " is detected none 68 landmark")
                continue
        else:
            lmks = lmks[0]
        # x_la, y_la, x_rb, y_rb = int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3])
        kpt = lmks[:, :2]
        tform = crop(image, lmks)
        # print(tform)  # 3x3
        # print(dense_lmk.shape)  # (478, 2)
        # print(lmks.shape)       # (68,2)
        # save crop image
        cropped_image = warp(image, tform.inverse, output_shape=(224, 224)) # 被归一化了
        cropped_image = cropped_image * 255
        
        # warp kpt
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T # np.linalg.inv(tform.params) 水平堆叠
        # cropped_kpt[:,:2] = cropped_kpt[:,:2]
        # cropped_kpt[:,:2] = cropped_kpt[:,:2]/cropped_image.shape[-1]*2 - 1
        
        # warp dense landmark
        cropped_dense_kpt = np.dot(tform.params, np.hstack([dense_lmk, np.ones([dense_lmk.shape[0], 1])]).T).T # np.linalg.inv(tform.params) 水平堆叠
        # cropped_dense_kpt[:,:2] = cropped_dense_kpt[:,:2]/cropped_image.shape[-1]*2 - 1
        # save landmark
        out_crop_whole_dir =  os.path.join(out_par_dir, real_path)
        lmk_path = os.path.join(lmk_dir, real_path, "lmk")
        dense_lmk_path = os.path.join(dense_lmk_dir, real_path, "dense_lmk")
        tform_path = os.path.join(tform_dir, real_path, "tform")
        
        if not os.path.exists(out_crop_whole_dir):
            os.makedirs(out_crop_whole_dir,exist_ok=True)
        if not os.path.exists(lmk_path):
            os.makedirs(lmk_path, exist_ok=True)
        if not os.path.exists(dense_lmk_path):
            os.makedirs(dense_lmk_path, exist_ok=True)
        if not os.path.exists(tform_path):
            os.makedirs(tform_path, exist_ok=True)
        
        cv2.imwrite(os.path.join(out_crop_whole_dir, image_basename + ".png"), cropped_image)
        np.save(os.path.join(lmk_path, image_basename + ".npy"), cropped_kpt)
        np.save(os.path.join(dense_lmk_path, image_basename + ".npy"), cropped_dense_kpt)
        np.save(os.path.join(tform_path, image_basename + ".npy"), tform)
        
    print("finish save data")

if __name__=="__main__":
    image_source_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/"
    lmk_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/crop_resized_224"
    dense_lmk_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/crop_resized_224"
    tform_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/crop_resized_224"
    generate_crop_image(image_source_dir, lmk_dir, dense_lmk_dir, tform_dir)