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
    # 定义模型路径
    det_model_path = '/ch_PP-OCRv4_det_infer'  # 文本检测模型路径
    rec_model_path = '/ch_PP-OCRv4_rec_infer'  # 文本识别模型路径
    cls_model_path = '/ch_ppocr_mobile_v2.0_cls_infer'  # 文字方向分类模型路径（如果使用）

    # 初始化PaddleOCR，指定自定义模型路径
    ocr = PaddleOCR(use_angle_cls=True,
                    lang='ch',  # 根据需要选择语言
                    det_model_dir=det_model_path,
                    rec_model_dir=rec_model_path,
                    cls_model_dir=cls_model_path)  # 如果不使用角度分类器，可以省略cls_model_dir参数

    # 图片路径
    # img = cv2.imread(img_path)
    # show(img, 'img')
    # 执行OCR
    result = ocr.ocr(img_path, cls=True)
    ans = []
    for line in result:
            if line is not None:
                res = extract_tuples(line)
                ans.append(res)
    return ans