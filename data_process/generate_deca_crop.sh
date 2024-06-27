nohup python generate_deca_crop_loop.py > generate_data_2024-6-25-3.log 2>&1 &  # 中途报错 、

# face alignment对于太小的人脸检测不出来，需要做异常捕捉
nohup python generate_deca_crop_loop.py > generate_data_2024-6-25-11.log  2>&1 & 

# 上面范了一个错误，所有的FAN的landmark都检测不到  image.shape[0] or image.shape[1] <= 50:
nohup python generate_deca_crop_loop.py > generate_data_2024-6-25-12.log  2>&1 & 


# 批量处理数据
nohup python generate_deca_crop_loop_gpu1.py > generate_data_gpu1_2024-6-25-15.log  2>&1 & 
nohup python generate_deca_crop_loop_gpu2.py > generate_data_gpu2_2024-6-25-15.log  2>&1 & 
nohup python generate_deca_crop_loop_gpu3.py > generate_data_gpu3_2024-6-25-15.log  2>&1 & 
nohup python generate_deca_crop_loop_gpu4.py > generate_data_gpu4_2024-6-25-15.log  2>&1 & 
nohup python generate_deca_crop_loop_gpu5.py > generate_data_gpu5_2024-6-25-15.log  2>&1 & 


nohup python generate_deca_crop_loop_FAN_gpu1.py > generate_data_gpu1_2024-6-25-24.log  2>&1 &  # [1] 1722694
nohup python generate_deca_crop_loop_FAN_gpu2.py > generate_data_gpu2_2024-6-25-24.log  2>&1 &  # [2] 1722851
nohup python generate_deca_crop_loop_FAN_gpu3.py > generate_data_gpu3_2024-6-25-24.log  2>&1 &  # [3] 1723037
nohup python generate_deca_crop_loop_FAN_gpu4.py > generate_data_gpu4_2024-6-25-24.log  2>&1 &  # [4] 1723493
nohup python generate_deca_crop_loop_FAN_gpu5.py > generate_data_gpu5_2024-6-25-24.log  2>&1 &  # [5] 1723815


nohup python generate_deca_crop_loop_FAN_gpu1.py > generate_data_gpu1_2024-6-26-10.log  2>&1 &  # [1] 1722694
nohup python generate_deca_crop_loop_FAN_gpu3.py > generate_data_gpu3_2024-6-26-10.log  2>&1 &  # [3] 1723037
nohup python generate_deca_crop_loop_FAN_gpu4.py > generate_data_gpu4_2024-6-26-10.log  2>&1 &  # [4] 1723493
nohup python generate_deca_crop_loop_FAN_gpu5.py > generate_data_gpu5_2024-6-26-10.log  2>&1 &  # [5] 1723815


nohup python generate_deca_crop_loop_FAN_gpu2.py > generate_data_gpu2_2024-6-27-11.log  2>&1 &
