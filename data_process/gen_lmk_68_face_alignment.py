import face_alignment
import torch
import os
from tqdm import tqdm
import cv2
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)


dataset_root = "/home/gdp/harddisk/Data1/VGGface2_for_DECA_finetune_224"
ldm_root = "/home/gdp/harddisk/Data1/VGGface2_for_DECA_finetune_dense_ldm_224_plus_oldman"

os.makedirs(ldm_root, exist_ok=True)
def gen_dataset(inpath = dataset_root, outpath=ldm_root):
    count = 0
    for path in tqdm(sorted(os.listdir(inpath)),"progress bar"): 
        #print(path)
        dir_name = os.path.join(inpath, path)
        
        out_dir = os.path.join(outpath, path)
        os.makedirs(out_dir, exist_ok=True)
        for root, _, files in os.walk(dir_name):
            for file in files:
                if file.lower().endswith('.png'):
                    image = file
                    img_name = image.split(".")[-2]
                    image_path = os.path.join(root, image)
                    # 在224上检测图片的landmark
                    lmks, _, bbox = face_detector.get_landmarks_from_image(image, return_landmark_score=True, return_bboxes=True) # 返回三项lmks、score、face bbox
                    # 根据bbox俩crop人脸
                    if lmks.any is None:  # 判断有问题
                        with open('exp_config_data/empty_dense_lmks_path-2024-6-24.txt', 'a') as f:
                            f.write(str(image_path) + '\n')
                        count+=1
                    # print("dense",dense_lmks)
                    npy_path = os.path.join(out_dir, img_name + ".npy")
                    np.save(npy_path, lmks, allow_pickle=True)
    print(count)

if __name__=="__main__":
    gen_dataset()

