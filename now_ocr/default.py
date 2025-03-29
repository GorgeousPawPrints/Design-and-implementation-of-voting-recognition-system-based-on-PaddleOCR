from paddleocr import PaddleOCR
import cv2

def show(img,name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_tuples(data):
    """
    递归提取嵌套列表中的所有元组。
    """
    result = []
    for item in data:
        if isinstance(item, tuple):  # 如果当前元素是元组
            result.append(item)
        elif isinstance(item, list):  # 如果当前元素是列表，递归处理
            result.extend(extract_tuples(item))
    return result

def begin(img_path):
    ocr = PaddleOCR(lang='ch') # need to run only once to load model into memory
    # img = cv2.imread(img_path)
    # show(img,'img')
    result = ocr.ocr(img_path)
    ans = []
    for line in result:
        if line is not None:
            res = extract_tuples(line)
            ans.append(res)
    return ans