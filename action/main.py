import os
import logging
from glob import glob
import cv2
from OpenCV import Preprocessing
from now_ocr.change import begin as first_begin
from now_ocr.default import begin as second_begin
import csv
from datetime import datetime
import shutil


# 配置日志记录器
def setup_logger():
    logger = logging.getLogger("OCRLogger")
    logger.setLevel(logging.DEBUG)

    log_file_handler = logging.FileHandler("ocr_log.txt", encoding="utf-8")
    log_file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file_handler.setFormatter(formatter)

    logger.addHandler(log_file_handler)
    logger.propagate = False
    return logger


def get_image_files(folder_path):
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.tiff', '*.webp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob(os.path.join(folder_path, ext), recursive=True))
        image_files.extend(glob(os.path.join(folder_path, '**', ext), recursive=True))
    return image_files


def clear_folder(folder_path):
    logger = setup_logger()

    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        logger.info(f"文件夹 {folder_path} 不存在！")
        return

    # 遍历文件夹中的所有内容
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)  # 获取完整路径
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                # 如果是文件或符号链接，直接删除
                os.unlink(item_path)
                logger.info(f"已删除文件: {item_path}")
            elif os.path.isdir(item_path):
                # 如果是子文件夹，递归删除
                shutil.rmtree(item_path)
                logger.info(f"已删除文件: {item_path}")
        except Exception as e:
            logger.info(f"无法删除 {item_path}: {e}")


def save_csv(need_save):
    # 定义 CSV 文件路径
    csv_file_path = "../Web/uploads/information.csv"

    # 初始化一个字典来存储单词及其计数
    word_count = {}

    # 如果 CSV 文件已存在，读取其中的数据
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 2:  # 确保每行有两列
                    word, count = row
                    word_count[word] = int(count)

    # 更新单词计数
    for word_set in need_save:
        for word in word_set:
            word_count[word] = word_count.get(word, 0) + 1

    # 按计数降序排序（新增的关键步骤）
    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)

    # 将排序后的数据写入 CSV 文件
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头（可选）
        writer.writerow(['Word', 'Count'])
        # 写入排序后的数据
        writer.writerows(sorted_words)


def action():
    logger = setup_logger()
    ppocr_logger = logging.getLogger("ppocr")
    ppocr_logger.setLevel(logging.ERROR)

    # 定义基础路径
    base_folder = r"D:\Program\Graduation Project\user_data"

    # 创建以当前时间命名的文件夹
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # 格式化时间戳
    new_folder_path = os.path.join(base_folder, current_time)  # 新文件夹路径
    os.makedirs(new_folder_path, exist_ok=True)  # 确保文件夹存在

    # 获取图片文件
    folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Web", "uploads")
    folder_path = os.path.abspath(folder_path)  # 转换为绝对路径
    os.makedirs(folder_path, exist_ok=True)  # 确保文件夹存在

    images = get_image_files(folder_path)
    images = list(set(images))
    need_save = []

    logger.info("找到的图片文件：")
    print(images)

    for img_path in images:
        logger.info(f"处理图片: {img_path}")

        try:
            # 图片预处理
            modified_image = Preprocessing.first_opt(img_path)

            # 打印路径以确认
            print(f"Processing image at: {img_path}")

            # 构造新的文件名（添加前缀 "new_"）
            original_file_name = os.path.basename(img_path)
            new_file_name = f"new_{original_file_name}"
            new_file_path = os.path.join(new_folder_path, new_file_name).replace("/", "\\")

            print(f"Saving processed image to: {new_file_path}")
            success = cv2.imwrite(new_file_path, modified_image)
            if success:
                logger.info(f"图片预处理完成并保存为: {new_file_path}")
            else:
                logger.error(f"Failed to save processed image: {new_file_path}")

            # 第一次 OCR 处理
            first_img = first_begin(new_file_path)
            logger.info(f"第一次 OCR 结果: {first_img}")

            # 第二次 OCR 处理
            second_img = second_begin(new_file_path)
            logger.info(f"第二次 OCR 结果: {second_img}")

        except Exception as e:
            logger.error(f"处理图片时出错: {img_path}, 错误信息: {str(e)}")
            continue

        combined_data = first_img + second_img
        if combined_data is None or len(combined_data) == 0:
            pass
        else:
            print(combined_data)
            unique_names = {item[0] for sublist in combined_data for item in sublist}
            need_save.append(unique_names)

    logger.info(f"当前处理后返回的识别结果: {need_save}")
    return need_save


if __name__ == '__main__':
    save_csv(action())