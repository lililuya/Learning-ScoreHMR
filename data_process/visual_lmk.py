 
import cv2
import torch
import numpy as np
import os
import torchvision
import torch.nn.functional as F

# boarder_points = [10, 109, 67, 103, 54, 21, 162, 127, 234, 93, 132, 58, 
#                    172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 
#                    365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 
#                    297, 338]

# related_point = [63, 107, 336, 293, 33, 133, 362, 263, 98, 327, 61, 
#                  291, 17, 152]


landmark_points_68 = [162,234,93,58,172,136,149,148,152,377,378,365,397,288,323,454,389,71,63,105,66,107,336,
                  296,334,293,301,168,197,5,4,75,97,2,326,305,33,160,158,133,153,144,362,385,387,263,373,
                  380,61,39,37,0,267,269,291,405,314,17,84,181,78,82,13,312,308,317,14,87]


def plot_all_kpts(image, kpts, color='b'):
    # 1) 先设置颜色
    # 2) 复制数据
    if color == 'r':
        c = (0, 0, 255)
    elif color == 'g':
        c = (0, 255, 0)
    elif color == 'b':
        c = (255, 0, 0)
    elif color == 'p':
        c = (255, 100, 100)
    image = image.copy()
    kpts = kpts.copy()

    for i in range(kpts.shape[0]): # 遍历关键点数组
        st = kpts[i, :2]
        image = cv2.circle(image, (int(st[0]), int(st[1])), 1, c, 2)
        # image = cv2.putText(image, str(i), (int(st[0])+3, int(st[1])+3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, c, 1, cv2.LINE_AA)
    return image
 
def tensor_vis_landmarks(images, landmarks, color='g'):
    # B C H W  0~1
    # B N 2
    vis_landmarks = []
    images = images.cpu().numpy()
    predicted_landmarks = landmarks.detach().cpu().numpy()

    for i in range(images.shape[0]):  # 可以是多张图片
        image = images[i]
        image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]].copy()  
        image = (image * 255)  # 反归一化
        predicted_landmark = predicted_landmarks[i] 
        image_landmarks = plot_all_kpts(image, predicted_landmark, color)
        vis_landmarks.append(image_landmarks)

    vis_landmarks = np.stack(vis_landmarks)
    vis_landmarks = torch.from_numpy(
        vis_landmarks[:, :, :, [2, 1, 0]].transpose(0, 3, 1, 2)) / 255.  # , dtype=torch.float32)
    return vis_landmarks


def merge_views(views):
    grid = []
    for view in views:
        grid.append(np.concatenate(view, axis=2))
    grid = np.concatenate(grid, axis=1)
    return to_image(grid)


def to_image(img):
    img = (img.transpose(1, 2, 0) * 255)[:, :, [2, 1, 0]]
    img = np.minimum(np.maximum(img, 0), 255).astype(np.uint8)
    return img


if __name__=="__main__":
    # image  B C H W
    image_dir = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_generate_deca_crop"
    dens_ldm_dir = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_generate_additional_data/dense_lmk/"
    ldm_dir = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_generate_additional_data/lmk/"
    savefolder = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_draw/"
    frame_id = 2
    
    image_path = os.path.join(image_dir, "0002_01.png")
    dens_ldm_path = os.path.join(dens_ldm_dir, "0002_01.npy")
    ldm_path = os.path.join(ldm_dir, "0002_01.npy")
    
    
    # gt
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) /255. # H W C
    image_tensor = torch.from_numpy(image.transpose(2,0,1)).unsqueeze(0)
    img_resize = F.interpolate(image_tensor, size=(224, 224), mode='bilinear')
    
    # dense
    dense_ldm_arr = np.load(dens_ldm_path, allow_pickle=True)
    dense_ldm_arr_tensor = torch.from_numpy(dense_ldm_arr)
    
    landmarks_dense = dense_ldm_arr_tensor[:].unsqueeze(0)
    landmarks_dense_68 = dense_ldm_arr_tensor[landmark_points_68].unsqueeze(0)
    
    # 68
    landmarks68 = np.load(ldm_path, allow_pickle=True)
    landmarks68_tensor = torch.from_numpy(landmarks68)
    landmarks68 = landmarks68_tensor.unsqueeze(0)
    
    # ldm B N 2
    row = []
    final_views = []
    dense_lmk_68 = tensor_vis_landmarks(img_resize, landmarks_dense_68, color='b')
    dense_lmk = tensor_vis_landmarks(img_resize, landmarks_dense, color='g')
    lmk68 = tensor_vis_landmarks(img_resize, landmarks68, color='r')
    # save
    row.append(dense_lmk_68[0].cpu().numpy())
    row.append(dense_lmk[0].cpu().numpy())
    row.append(lmk68[0].cpu().numpy())
    
    final_views.append(row)
    final_views = merge_views(final_views)
    
    # ldm68 = tensor_vis_landmarks(gt_lmks, landmarks[:, :17, :], color='g')
    # grid = torchvision.utils.make_grid(image_tensor, nrow=2, normalize=True)
    
    cv2.imwrite('{}/{}.jpg'.format(savefolder, frame_id), final_views)
    