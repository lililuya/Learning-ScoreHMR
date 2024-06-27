from sixdrepnet import SixDRepNet
import cv2
from math import cos, sin
import numpy as np
import torch
# 在Python中，函数参数传递的是对象的引用，而不是对象本身的副本。这意味着函数内部对参数进行的修改会影响到传递的对象。

def adjust_value_in_batch(weights, input_tensor):
     # 如果权重全为零，则将除了第一个维度之后的所有维度置为零
    if weights.sum() == 0:
        input_tensor[:] = 0
        # print(input_tensor)
    else:
        # 计算非零权重对应的维度平均值
        # nonzero_input_mean = torch.mean(input_tensor[weights != 0], dim=0, keepdim=True)

        # 将权重为0的维度的输入调整为其他维度的平均值
        indices = torch.where(weights == 0)
        input_tensor[indices] = 0

    return input_tensor

def head_toward_angle(images, threshold):
    images = images
    model = SixDRepNet()
    pitch, yaw, roll = model.predict_batch(images)

    yaw = -yaw * torch.pi / 180  # 将yaw转换为弧度并应用负号
    pitch = pitch * torch.pi / 180  # 将pitch转换为弧度
    roll = roll * torch.pi / 180  # 将roll转换为弧度
    # print("yaw shape",yaw.shape) #  yaw shape torch.Size([8])   
       
    # 逐元素计算旋转矩阵 R
    
    R = torch.zeros((images.shape[0], 3, 3))
    
    R[:, 0, 0] = torch.cos(yaw) * torch.cos(roll)
    R[:, 0, 1] = torch.cos(pitch) * torch.sin(roll) + torch.cos(roll) * torch.sin(pitch) * torch.sin(yaw)
    R[:, 0, 2] = -torch.cos(yaw) * torch.sin(pitch)
    R[:, 1, 0] = -torch.sin(roll)
    R[:, 1, 1] = torch.cos(pitch) * torch.cos(roll)
    R[:, 1, 2] = torch.sin(pitch)
    R[:, 2, 0] = torch.sin(yaw) * torch.cos(roll)
    R[:, 2, 1] = torch.cos(pitch) * torch.sin(yaw) * torch.sin(roll) - torch.sin(pitch) * torch.cos(yaw)
    R[:, 2, 2] = torch.cos(pitch) * torch.cos(yaw)
    
    # print("R.shape",R.shape) # R.shape torch.Size([8, 3, 3]) 
    
    normal_vector = torch.tensor([0, 0, 1]).unsqueeze(0)
    normal_vector = normal_vector.repeat(images.shape[0], 1)  # 复制法向量以匹配batch的大小
    
    cos_angle = normal_vector * R[:,:,2]
    angle = torch.acos(cos_angle)
    angle_degrees = angle * 180 / torch.pi
    angle_degrees = angle_degrees[:,2] # 取第三列，也就是z轴的值
    
    # print("angle_degrees",angle_degrees) # angle_degrees tensor([42.7654, 79.8790, 48.4504,  5.3540,  6.5146, 40.0446, 26.9592,  7.3299])
    
    weights = torch.where(angle_degrees > threshold, 0, 1) 
    # print("weights",weights) # angle_degrees tensor([42.7654, 79.8790, 48.4504,  5.3540,  6.5146, 40.0446, 26.9592,  7.3299]) 

    # print(weights)
    
    return weights


if __name__ == "__main__":
    # weights = torch.tensor([0,0,0,0])
    # batch = torch.rand(4,3)
    # adjust_value_in_batch(weights, batch)
    # print(nonzero_input_mean.shape) torch.Size([1, 3])
    import torch
    tensor = torch.randn(8,3)
    # 假设你有一个形状为 (1, 8) 的张量
    weights = torch.tensor([1, 0, 1, 0, 0, 1, 0, 1], dtype=torch.bool)

    # 使用 torch.where 获取满足条件的索引
    indices = torch.where(weights == 0)
    tensor[indices] = 0
    print(tensor)
    # # 打印结果
    # print("满足条件的索引:", indices)
    # print(weights[indices])