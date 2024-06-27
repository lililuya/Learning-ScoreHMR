from PIL import Image, ImageDraw, ImageFont
import os
from tqdm import tqdm
import cv2

font_path = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/font/TangXianBinSong-2.otf"
new_font_size = 50
new_font = ImageFont.truetype(font_path, size=new_font_size)

"""根据目录的字典顺序匹配"""
def list_images(directory):
    return [os.path.join(directory, f) for f in sorted(os.listdir(directory)) if f.endswith('.jpg') or f.endswith(".png")]

def resize_image(image_list, new_size):
    image_resize_list=[]
    for image in os.listdir(image_list):
        image = cv2.imread(image)
        image_resize = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        image_resize_list.append(image_resize)
    return image_resize_list

def create_collage(source_dir, processed_dir1, processed_dir2, output_file, images_per_row=10, row_spacing=40, group_rows=3):
    source_images = list_images(source_dir)
    processed1_images = list_images(processed_dir1)
    processed2_images = list_images(processed_dir2)

    # 取三个目录中数量最少的，以保证匹配不出错
    min_length = min(len(source_images), len(processed1_images), len(processed2_images))
    # print(min_length)
    matched_images = [(source_images[i], processed1_images[i], processed2_images[i]) for i in range(min_length)]
    if matched_images:
        img_sample = Image.open(matched_images[0][0])  # 取第一个图
        width, height = img_sample.size
        img_sample.close()

        collage_width = images_per_row * width
        rows_needed = len(matched_images) // images_per_row + (1 if len(matched_images) % images_per_row else 0)
        collage_height = 3*(rows_needed * height + (rows_needed // group_rows) * row_spacing + (1 if rows_needed % group_rows else 0)*row_spacing)
        collage = Image.new('RGB', (collage_width, collage_height))
        x_offset = 0
        y_offset = 0

        with tqdm(total=len(matched_images)) as pbar:
            for index, (source_img, processed1_img, processed2_img) in enumerate(matched_images):
                for i, img_path in enumerate([source_img, processed1_img, processed2_img]):
                    img = Image.open(img_path)
                    draw = ImageDraw.Draw(img)
                    draw.text((10, height - 50), f"Image {index + 1}-{i + 1}", font=new_font, fill="white")
                    collage.paste(img, (x_offset, y_offset + i * height))
                    img.close()

                x_offset += width
                if (index + 1) % images_per_row == 0:
                    x_offset = 0
                    y_offset += 3 * height + row_spacing  # Adjust y_offset to account for 3 images
                pbar.update(1)
        collage.save(output_file)
        print("save")

folder1 = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/test_generate_deca_crop"  # 第一个文件夹路径
folder2 = "/mnt/hd3/liwen/DATA_for_HMR/VggFace2/source_data_train/n000002"  # 第二个文件夹路径
folder3 = "/mnt/hd_ssd/cxh/dataset/VGGFace2-HQ/FFHQ_align/n000002"
output_folder = "/mnt/hd3/liwen/DECA/data_process/exp_config_data/compare/com.png"  # 输出文件夹路径

create_collage(folder1, folder2, folder3,  output_folder)

# 示例用法

