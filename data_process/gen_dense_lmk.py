import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import cv2
import numpy as np
from torch import NoneType
import face_alignment
import torch
from face_detector import FaceDetector
from tqdm import tqdm
import torch.nn.functional as F


if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

# 模型初始化    
face_detector_mediapipe = FaceDetector('google')
face_detector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device)

dataset_root = ""
ldm_root = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/dense_lmk_vgg_224"

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
                    # dir_path = os.path.join(inpath, dir_name) # /**/**/n005841
                    img_name = file.split(".")[-2]
                    file_path = os.path.join(root, file)
                    img = cv2.imread(file_path)
                    # print(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
                    # print(img)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
                    # b c h w
                    img_resize = F.interpolate(img_tensor, size=(224, 224), mode='bicubic')
                    img_resize = img_resize.squeeze(0).permute(1, 2, 0).numpy()*255.
                    img_resize = img_resize.astype(np.uint8)

                    dense_lmks = face_detector_mediapipe.dense(img_resize)
                    # print(file_path)
                    # assert dense_lmks is  None , "关键点未检出"
                    if dense_lmks.any is None:  # 判断有问题
                        with open('empty_dense_lmks_path-2024-1-30.txt', 'a') as f:
                            f.write(str(file_path) + '\n')
                        count+=1
                    # print("dense",dense_lmks)
                    npy_path = os.path.join(out_dir, img_name+".npy")
                    np.save(npy_path, dense_lmks, allow_pickle=True)
    print(count)

if __name__=="__main__":
    gen_dataset()