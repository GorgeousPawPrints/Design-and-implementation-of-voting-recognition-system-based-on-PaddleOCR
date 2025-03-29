import cv2
import matplotlib.pyplot  as plt

def get_crop(image):
    """
    random crop
    """
    import random
    h, w, _ = image.shape
    top_min = 1
    top_max = 8
    top_crop = int(random.randint(top_min, top_max))
    top_crop = min(top_crop, h - 1)
    crop_img = image.copy()
    ratio = random.randint(0, 1)
    if ratio:
        crop_img = crop_img[top_crop:h, :, :]
    else:
        crop_img = crop_img[0:h - top_crop, :, :]
    return crop_img

if __name__ == '__main__':
    raw_img = cv2.imread("D:\\Program\\Graduation Project\\0.png")
    plt.figure()
    plt.subplot(2, 1, 1)
    # 可视化原图
    plt.imshow(raw_img)
    # 随机切割
    crop_img = get_crop(raw_img)
    plt.subplot(2, 1, 2)
    # 可视化增广图
    plt.imshow(crop_img)
    plt.show()