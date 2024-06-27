import numpy as np
from skimage.transform import estimate_transform, warp


# DECA对数据的crop的方式
def crop(image, kpt):
    # 一些初始值的定义
    image_size = 224
    scale_min = 1.4
    scale_max = 1.8

    scale = [scale_min, scale_max]
    trans_scale_ori = 0.
    image_size = 224
    
    left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
    top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

    h, w, _ = image.shape
    old_size = (right - left + bottom - top)/2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
    # translate center
    trans_scale = (np.random.rand(2)*2 -1) * trans_scale_ori
    center = center + trans_scale * old_size # 0.5

    scale = np.random.rand() * (scale[1] - scale[0]) + scale[0]
    size = int(old_size * scale)

    # crop image
    src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
    DST_PTS = np.array([[0,0], [0, image_size - 1], [image_size - 1, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    
    # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
    # # change kpt accordingly
    # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
    return tform