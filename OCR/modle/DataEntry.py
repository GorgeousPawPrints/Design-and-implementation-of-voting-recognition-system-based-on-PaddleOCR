import math
import cv2
import numpy as np
import matplotlib.pyplot  as plt
import random
from PIL import Image

# 定义一个函数 resize_norm_img，用于调整图像大小并进行归一化处理
# 参数 img 是输入的图像，image_shape 是目标图像的形状，包含通道数、高度和宽度
def resize_norm_img(img, image_shape):
    # 从 image_shape 中解包出通道数 imgC、高度 imgH 和宽度 imgW
    imgC, imgH, imgW = image_shape
    # 获取输入图像的高度
    h = img.shape[0] 
    # 获取输入图像的宽度
    w = img.shape[1] 
    # 计算图像的宽高比
    ratio = w / float(h)
    # 判断按照目标高度和宽高比计算得到的宽度向上取整后是否超过目标宽度
    if math.ceil(imgH  * ratio) > imgW:
        # 如果超过，则将调整后的宽度设置为目标宽度
        resized_w = imgW
    else:
        # 否则，将调整后的宽度设置为按照目标高度和宽高比计算得到的宽度向上取整的值
        resized_w = int(math.ceil(imgH  * ratio))
    # 使用 OpenCV 的 resize 函数将图像调整为指定的宽度和高度
    resized_image = cv2.resize(img,  (resized_w, imgH))
    # 将调整后的图像数据类型转换为 float32，以便后续进行归一化等数值计算
    resized_image = resized_image.astype('float32') 
    # 判断目标图像的通道数是否为 1
    if image_shape[0] == 1:
        # 如果通道数为 1，将图像像素值除以 255 进行归一化，使其范围在 0 到 1 之间
        resized_image = resized_image / 255
        # 在图像数组的第 0 维添加一个维度，以符合单通道图像的输入格式
        resized_image = resized_image[np.newaxis, :]
    else:
        # 如果通道数不为 1，将图像的通道维度进行转置，从 (高度, 宽度, 通道) 转换为 (通道, 高度, 宽度)
        resized_image = resized_image.transpose((2,  0, 1))
        # 将图像像素值除以 255 进行归一化，使其范围在 0 到 1 之间
        resized_image = resized_image / 255
    # 对归一化后的图像进行零均值化处理，减去 0.5
    resized_image -= 0.5
    # 对零均值化后的图像进行标准化处理，除以 0.5
    resized_image /= 0.5
    # 创建一个全零的数组，形状为目标图像的形状，用于填充调整和归一化后的图像
    padding_im = np.zeros((imgC,  imgH, imgW), dtype=np.float32) 
    # 将调整和归一化后的图像填充到全零数组的对应位置
    padding_im[:, :, 0:resized_w] = resized_image
    # 返回填充后的图像
    return padding_im

def resize_norm_img_chinese(img, image_shape):
    imgC, imgH, imgW = image_shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im

# 主程序入口
if __name__ == '__main__':
    # 使用 OpenCV 的 imread 函数读取指定路径下的图像
    raw_img = cv2.imread("../0.png")
    # 创建一个新的图形窗口，用于显示图像
    plt.figure() 
    # 将图形窗口划分为 2 行 1 列的子图布局，并选择第 1 个子图
    plt.subplot(2,  1, 1)
    # 在第 1 个子图中可视化原始图像
    plt.imshow(raw_img) 
    # 调用 resize_norm_img 函数对原始图像进行缩放并归一化处理
    # 注意：原代码此处存在错误，resize_norm_img 函数只返回一个值，不能同时赋值给两个变量
    # 正确的调用方式应该是 padding_im = resize_norm_img(raw_img, (3, raw_img.shape[0],  raw_img.shape[1])) 
    padding_im = resize_norm_img(raw_img, (3, raw_img.shape[0],  raw_img.shape[1])) 
    # 将第 2 个子图设置为当前子图
    plt.subplot(2,  1, 2)
    # 由于 padding_im 的形状为 (通道, 高度, 宽度)，需要将其转换为 (高度, 宽度, 通道) 才能使用 imshow 显示
    draw_img = padding_im.transpose((1,  2, 0))
    # 对 draw_img 进行反归一化处理，以便正确显示图像
    draw_img = (draw_img * 0.5 + 0.5) * 255
    draw_img = draw_img.astype(np.uint8) 
    # 在第 2 个子图中可视化经过处理后的图像
    plt.imshow(draw_img) 
    # 显示图形窗口
    plt.show() 