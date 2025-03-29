from paddle import nn
from head import get_para_bias_attr


# Im2Seq: 将多通道图像特征图转换为1维序列特征
class Im2Seq(nn.Layer):
    def __init__(self, in_channels, ** kwargs):
        """
        图像特征转换为序列特征
        :param in_channels: 输入特征图的通道数
        """
        super().__init__()
        self.out_channels = in_channels  # 输出通道数保持不变

    def forward(self, x):
        """
        前向传播逻辑
        :param x: 输入特征图 (B, C, H, W)
        :return: 转换后的序列特征 (B, W, C)
        """
        B, C, H, W = x.shape
        assert H == 1, "输入特征图高度必须为1"  # 保证输入是单行文本特征图
        x = x.squeeze(axis=2)  # 去除高度维度 (B, C, W)
        x = x.transpose([0, 2, 1])  # 转换为 (B, W, C) 以适应序列模型输入格式
        return x


# EncoderWithRNN: 使用双向LSTM进行序列编码
class EncoderWithRNN(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        """
        初始化双向LSTM编码器
        :param in_channels: 输入特征维度
        :param hidden_size: RNN隐藏层大小
        """
        super().__init__()
        self.out_channels = hidden_size * 2  # 双向输出合并
        self.lstm = nn.LSTM(
            in_channels,  # 输入特征维度
            hidden_size,  # 隐藏层大小
            direction='bidirectional',  # 双向传播
            num_layers=2  # 双层LSTM
        )

    def forward(self, x):
        """
        前向传播逻辑
        :param x: 输入序列特征 (B, T, C)
        :return: 编码后特征 (B, T, 2H)
        """
        print(x.shape)
        x, _ = self.lstm(x)  # (B, T, 2H), (B, 2H)
        return x



# EncoderWithFC: 使用全连接层进行序列编码
class EncoderWithFC(nn.Layer):
    def __init__(self, in_channels, hidden_size):
        """
        初始化全连接层编码器
        :param in_channels: 输入特征维度
        :param hidden_size: 全连接层输出维度
        """
        super().__init__()
        self.out_channels = hidden_size

        # 获取权重和偏置的参数属性（带L2正则化）
        weight_attr, bias_attr = get_para_bias_attr(
            l2_decay=0.00001,  # L2正则化系数
            k=in_channels  # 正则化作用于输入权重
        )

        self.fc = nn.Linear(
            in_channels,  # 输入维度
            hidden_size,  # 输出维度
            weight_attr=weight_attr,  # 权重参数属性
            bias_attr=bias_attr,  # 偏置参数属性
            name='reduce_encoder_fea'  # 层名称
        )

    def forward(self, x):
        """
        前向传播逻辑
        :param x: 输入序列特征 (B, T, C)
        :return: 全连接层输出 (B, T, H)
        """
        x = self.fc(x)
        return x


# SequenceEncoder: 动态选择序列编码方式
class SequenceEncoder(nn.Layer):
    def __init__(self, in_channels, encoder_type, hidden_size=48, ** kwargs):
        """
        初始化序列编码器
        :param in_channels: 输入特征通道数
        :param encoder_type: 编码器类型 {'reshape', 'fc', 'rnn'}
        :param hidden_size: 隐藏层维度（仅fc/rnn有效）
        """
        super().__init__()
        self.encoder_reshape = Im2Seq(in_channels)  # 必选的形状转换层

        self.out_channels = self.encoder_reshape.out_channels  # 初始输出通道

        # 根据encoder_type选择不同的编码器实现
        if encoder_type == 'reshape':
            self.only_reshape = True  # 仅进行形状转换，无额外编码
        else:
            support_encoder_dict = {
                'reshape': Im2Seq,
                'fc': EncoderWithFC,
                'rnn': EncoderWithRNN
            }
            assert encoder_type in support_encoder_dict, \
                f"不支持的编码器类型: {encoder_type}, 支持类型: {support_encoder_dict.keys()}"

            # 动态创建对应的编码器实例
            self.encoder = support_encoder_dict[encoder_type](
                self.encoder_reshape.out_channels,  # 输入通道数
                hidden_size  # 输出维度
            )
            self.out_channels = self.encoder.out_channels  # 更新输出通道
            self.only_reshape = False

    def forward(self, x):
        """
        前向传播逻辑
        :param x: 输入特征图 (B, C, H, W)
        :return: 编码后的序列特征
        """
        x = self.encoder_reshape(x)  # 强制进行形状转换

        if not self.only_reshape:
            x = self.encoder(x)  # 执行额外编码（FC/RNN）
        return x