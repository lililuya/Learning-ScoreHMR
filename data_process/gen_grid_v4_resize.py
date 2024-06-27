from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm

def list_images(directory):
    """
    返回一个目录中所有图像文件的路径列表。
    """
    images = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    images = [f for f in images if f.endswith(".jpg") or f.endswith(".png")]  # 这里假设只有jpg和png格式的图片
    return sorted(images)

def resize_image(image_path, target_size):
    """
    调整图片大小至目标尺寸。
    """
    img = Image.open(image_path)
    img = img.resize(target_size, Image.ANTIALIAS)  # 使用抗锯齿缩放
    return img

def create_collage(source_dir, processed_dir1, processed_dir2, output_file, images_per_row=10, row_spacing=40, group_rows=3):
    source_images = list_images(source_dir)
    processed1_images = list_images(processed_dir1)
    processed2_images = list_images(processed_dir2)

    # 取三个目录中数量最少的，以保证匹配不出错
    min_length = min(len(source_images), len(processed1_images), len(processed2_images))
    matched_images = [(source_images[i], processed1_images[i], processed2_images[i]) for i in range(min_length)]
    if matched_images:
        # 统一调整图片大小为256x256
        target_size = (256, 256)
        collage_width = images_per_row * target_size[0]
        rows_needed = len(matched_images) // images_per_row + (1 if len(matched_images) % images_per_row else 0)
        collage_height = 3 * (rows_needed * target_size[1] + (rows_needed // group_rows) * row_spacing + (1 if rows_needed % group_rows else 0) * row_spacing)
        collage = Image.new('RGB', (collage_width, collage_height))
        x_offset = 0
        y_offset = 0

        with tqdm(total=len(matched_images)) as pbar:
            for index, (source_img, processed1_img, processed2_img) in enumerate(matched_images):
                for i, img_path in enumerate([source_img, processed1_img, processed2_img]):
                    # 调整图片大小
                    resized_img = resize_image(img_path, target_size)
                    draw = ImageDraw.Draw(resized_img)
                    draw.text((10, target_size[1] - 50), f"Image {index + 1}-{i + 1}", fill="white")
                    collage.paste(resized_img, (x_offset, y_offset + i * target_size[1]))
                    resized_img.close()

                x_offset += target_size[0]
                if (index + 1) % images_per_row == 0:
                    x_offset = 0
                    y_offset += 3 * target_size[1] + row_spacing  # 调整y_offset以适应3张图片的高度
                pbar.update(1)
        collage.save(output_file)
        print("保存成功")

# 设置文件夹路径和输出路径
folder1 = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n000002"
folder2 = "/mnt/hd_ssd/cxh/dataset/VGGFace2-HQ/FFHQ_align/n000002"
folder3 = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_generate_deca_crop"
output_folder = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/compare/com.png"

# 创建拼贴图
create_collage(folder1, folder2, folder3, output_folder)
