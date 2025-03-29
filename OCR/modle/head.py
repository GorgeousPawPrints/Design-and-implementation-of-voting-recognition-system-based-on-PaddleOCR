import math
import paddle
from paddle import ParamAttr, nn
from paddle.nn import functional as F
import numpy as np

def get_para_bias_attr(l2_decay, k):
    """
    获取权重和偏置的参数属性（含L2正则化和均匀初始化）

    Args:
        l2_decay (float): L2正则化系数，防止过拟合
        k (int): 输入参数的维度，用于计算初始化标准差

    Returns:
        list[ParamAttr, ParamAttr]:
            包含权重参数属性和偏置参数属性的列表
    """
    regularizer = paddle.regularizer.L2Decay(l2_decay)  # 创建L2正则化器
    stdv = 1.0 / math.sqrt(k * 1.0)  # 根据输入维度计算标准差
    initializer = nn.initializer.Uniform(-stdv, stdv)  # 定义均匀分布初始化器

    weight_attr = ParamAttr(regularizer=regularizer, initializer=initializer)  # 权重参数属性
    bias_attr = ParamAttr(regularizer=regularizer, initializer=initializer)  # 偏置参数属性
    return [weight_attr, bias_attr]  # 返回参数属性组合


class CTCHead(nn.Layer):
    """
    CTC损失头部网络，用于序列分类任务（如OCR文字识别）

    Args:
        in_channels (int): 输入特征维度
        out_channels (int): 输出类别数
        fc_decay (float): 全连接层权重衰减系数（默认0.0004）
        mid_channels (int, optional): 中间层维度（可选，默认None）
    """

    def __init__(self, in_channels, out_channels, fc_decay=0.0004, mid_channels=None, ** kwargs):
        super(CTCHead, self).__init__()

        self.out_channels = out_channels  # 存储输出通道数
        self.mid_channels = mid_channels  # 存储中间层维度

        # 根据是否存在中间层构建网络结构
        if mid_channels is None:
            # 使用单层全连接
            weight_attr, bias_attr = get_para_bias_attr(l2_decay=fc_decay, k=in_channels)
            self.fc = nn.Linear(
                in_channels,  # 输入维度
                out_channels,  # 输出维度
                weight_attr=weight_attr,  # 权重参数属性
                bias_attr=bias_attr  # 偏置参数属性
            )
        else:
            # 使用双层全连接
            # 第一层：输入层到中间层
            weight_attr1, bias_attr1 = get_para_bias_attr(l2_decay=fc_decay, k=in_channels)
            self.fc1 = nn.Linear(
                in_channels,
                mid_channels,
                weight_attr=weight_attr1,
                bias_attr=bias_attr1
            )

            # 第二层：中间层到输出层
            weight_attr2, bias_attr2 = get_para_bias_attr(l2_decay=fc_decay, k=mid_channels)
            self.fc2 = nn.Linear(
                mid_channels,
                out_channels,
                weight_attr=weight_attr2,
                bias_attr=bias_attr2
            )

    def forward(self, x, targets=None):
        """
        前向传播函数

        Args:
            x (Tensor): 输入特征图（形状[B, T, C]）
            targets (Tensor, optional): 真实标签（仅在训练时使用）

        Returns:
            Tensor: 网络预测结果
                - 训练模式：未应用softmax的原始预测值（形状[B, T, C]）
                - 推理模式：应用softmax后的概率分布（形状[B, T, C]）
        """
        if self.mid_channels is None:
            predicts = self.fc(x)  # 单层全连接
        else:
            predicts = self.fc1(x)  # 第一层全连接
            predicts = self.fc2(predicts)  # 第二层全连接

        # 推理模式下应用softmax归一化
        if not self.training:
            predicts = F.softmax(predicts, axis=2)  # 沿时间步维度归一化

        return predicts

def decode(text_index, text_prob=None, is_remove_duplicate=False):
    """ convert text-index into text-label. """

    character = "-0123456789abcdefghijklmnopqrstuvwxyz"
    result_list = []
    # 忽略tokens [0] 代表ctc中的blank位
    ignored_tokens = [0]
    batch_size = len(text_index)
    for batch_idx in range(batch_size):
        char_list = []
    conf_list = []
    for idx in range(len(text_index[batch_idx])):
        if text_index[batch_idx][idx] in ignored_tokens:
            continue
        # 合并blank之间相同的字符
        if is_remove_duplicate:
            # only for predict
            if idx > 0 and text_index[batch_idx][idx - 1] == text_index[
                batch_idx][idx]:
                continue
        # 将解码结果存在char_list内
        char_list.append(character[int(text_index[batch_idx][
                                           idx])])
        # 记录置信度
        if text_prob is not None:
            conf_list.append(text_prob[batch_idx][idx])
        else:
            conf_list.append(1)
    text = ''.join(char_list)
    # 输出结果
    result_list.append((text, np.mean(conf_list)))
    return result_list