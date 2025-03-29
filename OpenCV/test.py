import cv2
import numpy as np

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

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    orig = img.copy()
    img = resize(img, width=600)
    ratio = img.shape[0] / orig.shape[0]  # 原图高度 : 缩放后高度
    scale_factor = 1.0 / ratio if ratio != 0 else 1.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 50, 150)

    # 展示预处理结果
    print("STEP 1: 边缘检测")
    show(img, "img")
    show(edged, "edged")

    # 轮廓检测
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 根据面积筛选轮廓（可选）
    min_area = 10  # 根据实际情况调整阈值
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
    corners=best_contour=[]
    for cnt in filtered_contours:
        # 提取边界矩形
        x, y, w, h = cv2.boundingRect(cnt)
        real_x = int(x * scale_factor)
        real_y = int(y * scale_factor)
        real_w = int(w * scale_factor)
        real_h = int(h * scale_factor)
        cv2.rectangle(orig, (real_x, real_y), (real_x + real_w, real_y + real_h), (0, 255, 0), 2)
        # 计算四个角点
        corners = [
            (x, y),
            (x + w, y),
            (x + w, y + h),
            (x, y + h)
        ]
        # 映射回原图坐标系
        real_corners = [
            (int(x * scale_factor), int(y * scale_factor)),
            (int((x + w) * scale_factor), int(y * scale_factor)),
            (int((x + w) * scale_factor), int((y + h) * scale_factor)),
            (int(x * scale_factor), int((y + h) * scale_factor))
        ]
        best_contour = (corners, real_corners)
        break  # 只取第一个满足条件的轮廓


    # 展示结果
    print("STEP 2: 获取轮廓")
    orig = resize(orig, height=600)
    show(orig, "orig")

    if not best_contour:
        print("未检测到有效轮廓")
        return orig

    # 提取源点和目标点
    src, real_src = best_contour
    dst = np.array([
        [500, 500],  # 左上角
        [1500, 500],  # 右上角
        [1500, 1500],  # 右下角
        [500, 1500]  # 左下角
    ], dtype=np.float32)
    src = np.array(real_src, dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(orig, M, (3000, 3000))

    # 二值处理
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(warped, 100, 255, cv2.THRESH_BINARY_INV)
    show(warped, "warped")
    show(binary, "binary")
    show(orig, "warped")

if __name__ == "__main__":
    input_path = "../credit_card_02.png"
    # ../2
    processed_image = preprocess_image(input_path)
