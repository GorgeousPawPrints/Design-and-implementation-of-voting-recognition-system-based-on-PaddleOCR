import paddle
from paddle import nn
import paddle.nn.functional as F
import cv2
from DataEntry import resize_norm_img


def make_divisible(v, divisor=8, min_value=None):
    """
    确保被8整除
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Layer):
    def __init__(self,
                 in_channels=3,
                 model_name='large',
                 scale=0.5,
                 disable_se=False,
                 **kwargs):
        super(MobileNetV3, self).__init__()
        self.disable_se = disable_se

        if model_name == "large":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, 'relu', 1],
                [3, 64, 24, False, 'relu', 2],
                [3, 72, 24, False, 'relu', 1],
                [5, 72, 40, True, 'relu', 2],
                [5, 120, 40, True, 'relu', 1],
                [5, 120, 40, True, 'relu', 1],
                [3, 240, 80, False, 'hardswish', 2],
                [3, 200, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 184, 80, False, 'hardswish', 1],
                [3, 480, 112, True, 'hardswish', 1],
                [3, 672, 112, True, 'hardswish', 1],
                [5, 672, 160, True, 'hardswish', 2],
                [5, 960, 160, True, 'hardswish', 1],
                [5, 960, 160, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 960
        elif model_name == "small":
            cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, 'relu', 2],
                [3, 72, 24, False, 'relu', 2],
                [3, 88, 24, False, 'relu', 1],
                [5, 96, 40, True, 'hardswish', 2],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 240, 40, True, 'hardswish', 1],
                [5, 120, 48, True, 'hardswish', 1],
                [5, 144, 48, True, 'hardswish', 1],
                [5, 288, 96, True, 'hardswish', 2],
                [5, 576, 96, True, 'hardswish', 1],
                [5, 576, 96, True, 'hardswish', 1],
            ]
            cls_ch_squeeze = 576
        else:
            raise NotImplementedError("mode[" + model_name +
                                      "_model] is not implemented!")

        supported_scale = [0.35, 0.5, 0.75, 1.0, 1.25]
        assert scale in supported_scale, \
            "supported scale are {} but input scale is {}".format(supported_scale, scale)
        inplanes = 16
        # conv1
        self.conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            if_act=True,
            act='hardswish')

        self.stages = []
        self.out_channels = []
        block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in cfg:
            se = se and not self.disable_se
            start_idx = 2 if model_name == 'large' else 0
            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []
            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    use_se=se,
                    act=nl))
            inplanes = make_divisible(scale * c)
            i += 1
        block_list.append(
            ConvBNLayer(
                in_channels=inplanes,
                out_channels=make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                if_act=True,
                act='hardswish'))
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))
        for i, stage in enumerate(self.stages):
            self.add_sublayer(sublayer=stage, name="stage{}".format(i))

    def forward(self, x):
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups=1,
                 if_act=True,
                 act=None):
        """
        卷积BN层
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核尺寸
        :parma stride: 步长大小
        :param padding: 填充大小
        :param groups: 二维卷积层的组数
        :param if_act: 是否添加激活函数
        :param act: 激活函数
        """
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False)

        self.bn = nn.BatchNorm(num_channels=out_channels, act=None)

    def forward(self, x):
        # conv层
        x = self.conv(x)
        # batchnorm层
        x = self.bn(x)
        # 是否使用激活函数
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 use_se,
                 act=None):
        """
        残差层
        :param in_channels: 输入通道数
        :param mid_channels: 中间通道数
        :param out_channels: 输出通道数
        :param kernel_size: 卷积核尺寸
        :parma stride: 步长大小
        :param use_se: 是否使用se模块
        :param act: 激活函数
        """
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.bottleneck_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=int((kernel_size - 1) // 2),
            groups=mid_channels,
            if_act=True,
            act=act)
        if self.if_se:
            self.mid_se = SEModule(mid_channels)
        self.linear_conv = ConvBNLayer(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.add(inputs, x)
        return x


class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4):
        """
        SE模块
        :param in_channels: 输入通道数
        :param reduction: 通道缩放率
        """
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        # 平均池化
        outputs = self.avg_pool(inputs)
        # 第一个卷积层
        outputs = self.conv1(outputs)
        # relu激活函数
        outputs = F.relu(outputs)
        # 第二个卷积层
        outputs = self.conv2(outputs)
        # hardsigmoid 激活函数
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs

if __name__ == '__main__':
    # 定义网络输入shape
    IMAGE_SHAPE_C = 3
    IMAGE_SHAPE_H = 32
    IMAGE_SHAPE_W = 320
    # 可视化网络结构
    paddle.summary(MobileNetV3(), [(1, IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W)])
    # 图片输入骨干网络
    backbone = MobileNetV3()
    # 将numpy数据转换为Tensor
    raw_img = cv2.imread("../0.png")
    padding_im = resize_norm_img(raw_img, (3, raw_img.shape[0],  raw_img.shape[1]))
    input_data = paddle.to_tensor([padding_im])
    # 骨干网络输出
    feature = backbone(input_data)
    # 查看feature map的纬度
    print("backbone output:", feature.shape)