from paddleocr import PaddleOCR
import cv2

def show(img,name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

ocr = PaddleOCR(lang='ch') # need to run only once to load model into memory
img_path = '../2.jpg'
img = cv2.imread(img_path)
show(img,'img')
result = ocr.ocr(img_path)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)