import os
import cv2
import torch
import torch.nn.functional as F
from tqdm import tqdm

dataset_root = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_train/"
resized_dataset_root = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/resized_224/"

os.makedirs(resized_dataset_root, exist_ok=True)

def gen_dataset(inpath=dataset_root, outpath=resized_dataset_root):
    for path in tqdm(sorted(os.listdir(inpath)), "Progress Bar"):
        dir_name = os.path.join(inpath, path)
        out_dir = os.path.join(outpath, path)
        os.makedirs(out_dir, exist_ok=True)
        
        for root, _, files in os.walk(dir_name):
            for file in files:
                if file.lower().endswith('.png'):
                    img_path = os.path.join(root, file)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    img = torch.from_numpy(img.transpose((2, 0, 1)))  # Convert to PyTorch tensor (C, H, W)
                    img = img.unsqueeze(0).float() / 255.0  # Add batch dimension and normalize to [0, 1]
                    img_resize = F.interpolate(img, size=(224, 224), mode='bicubic')
                    
                    # Clip the values to [0, 1]
                    img_resize = torch.clamp(img_resize, 0, 1)
                    
                    out_file = os.path.join(out_dir, file)
                    img_resize = img_resize.squeeze(0) * 255.0  # Remove batch dimension and scale back to [0, 255]
                    img_resize = img_resize.byte().numpy().transpose((1, 2, 0))  # Convert back to numpy array (H, W, C)
                    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
                    cv2.imwrite(out_file, img_resize)

if __name__ == "__main__":
    gen_dataset()