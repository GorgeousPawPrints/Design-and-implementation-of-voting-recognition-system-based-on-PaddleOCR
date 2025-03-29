from paddleocr import PaddleOCR
import cv2

def show(img,name):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
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
    img_path = '../2.jpg'
    img = cv2.imread(img_path)
    show(img, 'img')
    # 执行OCR
    result = ocr.ocr(img_path, cls=True)

    # 输出结果
    for line in result:
        print(line)