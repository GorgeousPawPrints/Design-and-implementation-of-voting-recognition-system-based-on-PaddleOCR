import cv2
import numpy as np
from paddle.distributed.fleet.fleet_executor_utils import origin


def show(img,name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

def find_edge(gray,sigma=0.33):
    # 计算单通道像素强度的中位数
    # Otsu算法
    v= np.median(gray)
    mean = np.mean(gray)
    std = np.std(gray)
    lower = int(mean - 0.33 * std)
    upper = int(mean + 0.33 * std)
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(gray, lower, upper)
    print("lower : ", lower, ", upper : ", upper)
    show(gray, "gray")
    show(edged, "edged")
    # 轮廓检测
    # 新版opencv是0位置存储轮廓
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    min_area = 100  # 根据实际情况调整阈值
    filtered_contours = [cnt for cnt in cnts if cv2.contourArea(cnt) >= min_area]
    for cnt in filtered_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    show(gray,"gray")
    # 一个图片中可能有多个票，取最大的五个轮廓
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # 遍历轮廓
    screenCnt = None
    for c in cnts:
        # 计算轮廓近似
        peri = cv2.arcLength(c, True)
        # c表示输入的点集
        # epsilon表示从原始轮廓到近似轮廓的最大距离，他是一个准确度参数，这里是长度的百分之2
        # True表示是封闭的
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        # 4个点的时候拿出来
        print(len(approx))
        if len(approx) == 4:
            screenCnt = approx
            break
    return screenCnt

def order_points(pts):
    # 一共4个坐标点
    rect = np.zeros((4, 2), dtype="float32")

    # 按顺序找到对应坐标0123分别是 左上，右上，右下，左下
    # 计算左上，右下
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 计算右上和左下
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def transform(image,pts):
    # 获取输入坐标
    rect = order_points(pts)
    # 左上 右上 右下 左下
    (tl, tr, br, bl) = rect
    # 计算输入的w和h值

    width_A = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_B = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    width = max(int(width_A), int(width_B))
    print(f"width = {width}")
    height_A = np.sqrt(((tr[0] - bl[0]) ** 2) + ((tr[1] - bl[1]) ** 2))
    heigth_A = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    height = max(int(height_A), int(heigth_A))
    print(f"height = {height}")

    # 变换后对应坐标位置
    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype="float32")

    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))

    return warped


def preprocess_image(image_path):
    # 读入图像
    img = cv2.imread(image_path)

    # 复制保存原始图像
    orig = img.copy()

    # 将原图像进行放缩
    img = resize(img, width=600)
    # 应用细节增强
    img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)

    # 计算保留原始图像进行放缩多少倍数
    ratio = img.shape[0] / orig.shape[0]  # 原图高度 : 缩放后高度
    scale_factor = 1.0 / ratio if ratio != 0 else 1.0
    # 获取图像的灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 增加对比度
    # gray = cv2.equalizeHist(gray)


    # 对灰度图进行高斯滤波
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # 展示预处理结果
    print("STEP 1: 边缘检测")
    show(img, "img")
    # 进行Canny检测轮廓


    screenCnt = find_edge(gray)
    if screenCnt is not None:  # 检查是否找到了符合条件的轮廓
        print("STEP 2")
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 2)
        show(img, "screenCnt")
    else:
        print("未找到有效的四边形轮廓")
        return orig

    warped = transform(orig, screenCnt.reshape(4, 2) / ratio)
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    show(warped, "warped")
    return warped


if __name__ == "__main__":
    input_path = "../jjj.jpg"
    # ../2
    processed_image = preprocess_image(input_path)
