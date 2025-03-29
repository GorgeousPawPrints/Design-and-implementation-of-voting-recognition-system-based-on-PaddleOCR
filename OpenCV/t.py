import cv2

# 读取图像
img = cv2.imread('../111.jpg', 0)

# 创建CLAHE对象并应用
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img)

# 显示结果
cv2.imshow('CLAHE', cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()