import paddle
import cv2
from DataEntry import resize_norm_img
from backbone import MobileNetV3
from neck import SequenceEncoder
from head import CTCHead
from paddle.nn import functional as F
from head import decode

if __name__ == '__main__':
    # # 定义网络输入shape
    # IMAGE_SHAPE_C = 3
    # IMAGE_SHAPE_H = 32
    # IMAGE_SHAPE_W = 320
    # # 可视化网络结构
    # paddle.summary(MobileNetV3(), [(1, IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W)])
    # # 图片输入骨干网络
    backbone = MobileNetV3()
    # # 将numpy数据转换为Tensor
    raw_img = cv2.imread("D:\\Program\\Graduation Project\\0.png")
    padding_im = resize_norm_img(raw_img, (3, raw_img.shape[0],  raw_img.shape[1]))
    input_data = paddle.to_tensor([padding_im])
    # # 骨干网络输出
    feature = backbone(input_data)[-1]
    print(feature.shape)
    # # 查看feature map的纬度
    # print("backbone output:", feature)
    neck = SequenceEncoder(in_channels=480,encoder_type='rnn')
    sequence = neck(feature)
    print("sequence shape:", sequence.shape)

    ctc_head = CTCHead(in_channels=96, out_channels=37)
    predict = ctc_head(sequence)
    print("predict shape:", predict.shape)
    result = F.softmax(predict, axis=2)
    pred_id = paddle.argmax(result, axis=2)
    pred_socres = paddle.max(result, axis=2)
    print("pred_id:", pred_id)
    print("pred_scores:", pred_socres)
    decode_out = decode(pred_id, pred_socres)
    print("decode out:", decode_out)
    right_pred_id = paddle.to_tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8]])  # 示例索引
    tmp_scores = paddle.to_tensor([[0.9] * 9])  # 对应概率
    out = decode(right_pred_id, tmp_scores)
    print("out:", out)