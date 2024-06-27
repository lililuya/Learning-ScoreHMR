# Import SixDRepNet
from sixdrepnet import SixDRepNet
import cv2
from math import cos, sin
import numpy as np

# Create model
# Weights are automatically downloaded
model = SixDRepNet()

# img = cv2.imread('/home/gdp/harddisk/Data1/VGGface2_for_DECA_finetune_224/n000002/0024_01.png') # 大正脸 10度
# img = cv2.imread('/home/gdp/harddisk/Data1/VGGface2_for_DECA_finetune_224/n000002/0330_01.png')  # 朝右的脸 50

img = cv2.imread('/home/gdp/harddisk/Data1/VGGface2_for_DECA_finetune_224/n000023/0325_01.png') #朝左的脸 60

pitch, yaw, roll = model.predict(img)
model.draw_axis(img, yaw, pitch, roll)
cv2.imwrite("headpose_toward_left_face.png", img)

yaw = -yaw * np.pi / 180  # 注意取负号
pitch = pitch * np.pi / 180
roll = roll * np.pi / 180

R = np.array([[cos(yaw) * cos(roll), cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw), -cos(yaw) * sin(pitch)],
              [-sin(roll), cos(pitch) * cos(roll), sin(pitch)],
              [sin(yaw) * cos(roll), cos(pitch) * sin(yaw) * sin(roll) - sin(pitch) * cos(yaw), cos(pitch) * cos(yaw)]])

normal_vector = np.array([0, 0, 1])

cos_angle = np.dot(normal_vector, R[:, 2])
angle = np.arccos(cos_angle)
angle_degrees = angle * 180 / np.pi

# 上面这段代码是求和出屏幕的法向量的夹角 

print(angle_degrees)




# print(yaw, pitch, roll)
# [-1.6712008] [-10.425742] [1.614235]


# cv2.imshow("test_window", img)
# cv2.waitKey(0)
