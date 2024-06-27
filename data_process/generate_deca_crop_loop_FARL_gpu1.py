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
from glob import glob
import argparse
from facer.face_alignment.network import denormalize_points, heatmap2points

# 设置日志等级
logging.basicConfig(level=logging.INFO)

device = "cuda:1" if torch.cuda.is_available() else "cpu"
face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)
logging.info("FAN detector load successful!")
face_detector_mediapipe = FaceDetector('google')
logging.info("Mediapipe detector load successful!")

backlist = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/black_list/gpu1/backlist_lmk.txt"
processed_path_list = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/black_list/gpu1/processed.txt"

def generate_crop_image(image_dir, lmk_dir, dense_lmk_dir, tform_dir, out_par_dir):
    image_list = sorted(glob(image_dir + '/*/*.jpg') +  glob(image_dir + '/*/*.png'))[:600000]
    
    if args.recovery:
        processed_paths = set()
        if os.path.exists(processed_path_list):
            with open(processed_path_list, 'r') as f:
                processed_paths = set(f.read().splitlines())
        image_list = sorted([image_path for image_path in image_list if image_path not in processed_paths])
                
    for image_path in tqdm(image_list):
        # 放在前面，检测到非法也算过了
        with open(processed_path_list, "a") as file:
            file.write(image_path + "\n")
        image_basename = os.path.basename(os.path.splitext(image_path)[0])
        real_path = os.path.relpath(os.path.dirname(image_path), image_dir) # 子目录
        image = cv2.imread(image_path)
        # 貌似有些请情况会出现异常
        try:
            lmks, _, bbox = face_detector.get_landmarks_from_image(image_path, return_landmark_score=True, return_bboxes=True) # 返回三项lmks、score、face bbox
            dense_lmk = face_detector_mediapipe.dense(image)
        except Exception as e:
            print(f"===== Error catch exception in : {type(e).__name__}, exceptio path: {image_path} =====")
            
        if dense_lmk is None or bbox is None:
            with open(backlist, "a") as file:
                file.write(image_path + "\n")
                print(image_path, " is detected no landmark")
                continue
            
        tform = crop(image, lmks)
    
        # save crop image
        cropped_image = warp(image, tform.inverse, output_shape=(224, 224)) # 被归一化了
        cropped_image = cropped_image * 255
        
        # warp kpt
        cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0], 1])]).T).T # np.linalg.inv(tform.params) 水平堆叠
        # warp dense landmark
        cropped_dense_kpt = np.dot(tform.params, np.hstack([dense_lmk, np.ones([dense_lmk.shape[0], 1])]).T).T # np.linalg.inv(tform.params) 水平堆叠
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
    parser = argparse.ArgumentParser(description='Generate data for deca, include lmk、dense_lmk、cropped images')
    parser.add_argument('--recovery', type=bool, required=False, default=False, help="Recovery from exception, skip the processed files")
    # parser.add_argument('--lmk_dir', type=str, required=False, default="", help="")
    # parser.add_argument('--dense_lmk_dir', type=str, required=False, default="", help="")
    # parser.add_argument('--tform_dir', type=str, required=False, default="", help="")
    # parser.add_argument('--out_cropped_dir', type=str, required=False, default="", help="")
    args = parser.parse_args()
    
    image_source_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/"
    lmk_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/crop_resized_224_lmk"
    dense_lmk_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/crop_resized_224_lmk"
    tform_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/crop_resized_224_lmk"
    out_cropped_dir = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/crop_resized_224"
    if not os.path.exists(image_source_dir):
        os.makedirs(image_source_dir, exist_ok=True)
    if not os.path.exists(lmk_dir):
        os.makedirs(lmk_dir, exist_ok=True)
    if not os.path.exists(dense_lmk_dir):
        os.makedirs(dense_lmk_dir, exist_ok=True)
    if not os.path.exists(tform_dir):
        os.makedirs(tform_dir, exist_ok=True)
    if not os.path.exists(out_cropped_dir):
        os.makedirs(out_cropped_dir, exist_ok=True)
    generate_crop_image(image_source_dir, lmk_dir, dense_lmk_dir, tform_dir, out_cropped_dir)