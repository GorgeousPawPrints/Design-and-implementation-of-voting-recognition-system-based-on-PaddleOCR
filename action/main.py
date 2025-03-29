import OpenCV.Preprocessing
import ppocr.test
import os
from glob import glob


def get_image_files(folder_path):
    # 定义支持的图片文件扩展名
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']

    # 获取所有匹配的图片文件路径
    image_files = []
    for ext in image_extensions:
        # 使用 glob 匹配当前扩展名的所有文件
        image_files.extend(glob(os.path.join(folder_path, ext), recursive=True))
        # 如果需要匹配子文件夹中的文件，可以使用 '**' 通配符
        image_files.extend(glob(os.path.join(folder_path, '**', ext), recursive=True))

    return image_files


# 示例用法
if __name__ == "__main__":
    folder_path = '../Web/uploads'  # 替换为目标文件夹路径
    images = get_image_files(folder_path)
    print("找到的图片文件：")
    for img in images:
        print(img)
