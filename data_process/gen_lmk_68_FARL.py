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
                    # dir_path = os.path.join(inpath, dir_name) # /**/**/n005841
                    img_name = file.split(".")[-2]
                    file_path = os.path.join(root, file)
                    img = cv2.imread(file_path)
                    # print(img)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.  # 规范化到0-1
                    # print(img)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(torch.float32)
                    # b c h w
                    img_resize = F.interpolate(img_tensor, size=(224, 224), mode='bicubic')  # 这个插值如果已经到了224应该不影响
                    img_resize = img_resize.squeeze(0).permute(1, 2, 0).numpy()*255.
                    img_resize = img_resize.astype(np.uint8)

                    lmks, _, detected_faces = face_detector.get_landmarks_from_image(img_resize, return_landmark_score=True, return_bboxes=True)
                    # print(file_path)
                    # assert dense_lmks is  None , "关键点未检出"
                    if img_resize.any is None:  # 判断有问题
                        with open('exp_config_data/empty_dense_lmks_path-2024-6-24.txt', 'a') as f:
                            f.write(str(file_path) + '\n')
                        count+=1
                    # print("dense",dense_lmks)
                    npy_path = os.path.join(out_dir, img_name + ".npy")
                    np.save(npy_path, img_resize, allow_pickle=True)
    print(count)

if __name__=="__main__":
    gen_dataset()

