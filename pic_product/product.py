import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

# 定义要随机生成的符号列表
symbols = ['√', '×']


# 生成指定符号图像并应用随机变换
def generate_symbol(symbol_char):
    image_size = (100, 100)
    img = Image.new('RGB', image_size,
                    (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255)))  # 随机背景颜色
    draw = ImageDraw.Draw(img)

    # 使用字体渲染符号
    try:
        font = ImageFont.truetype("arial.ttf", random.randint(30, 50))  # 随机字体大小
    except IOError:
        print("Arial字体文件未找到，请检查路径或安装字体")
        return None

    # 设置文本颜色为随机颜色
    color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))

    # 获取文本尺寸以确定位置（使用textbbox来代替已弃用的textsize）
    left, top, right, bottom = draw.textbbox((0, 0), symbol_char, font=font)
    text_width = right - left
    text_height = bottom - top

    # 随机位置偏移
    position_offset_x = random.randint(-10, 10)
    position_offset_y = random.randint(-10, 10)
    position = (
    (image_size[0] - text_width) // 2 + position_offset_x, (image_size[1] - text_height) // 2 + position_offset_y)

    # 绘制符号
    draw.text(position, symbol_char, fill=color, font=font)

    # 随机旋转角度
    rotation_angle = random.randint(-15, 15)
    img = img.rotate(rotation_angle, expand=True)

    # 添加随机比例缩放
    scale_factor = random.uniform(0.8, 1.2)
    new_size = (int(image_size[0] * scale_factor), int(image_size[1] * scale_factor))
    img = img.resize(new_size, Image.Resampling.LANCZOS)  # 使用新的Resampling方法

    # 应用随机仿射变换（例如：倾斜或拉伸）
    shear_x = random.uniform(-0.2, 0.2)
    shear_y = random.uniform(-0.2, 0.2)
    img = img.transform(image_size, Image.AFFINE, (1, shear_x, -position_offset_x, shear_y, 1, -position_offset_y))

    # 添加随机高斯模糊
    blur_radius = random.uniform(0, 1.5)
    img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


# 确保输出目录存在
output_dir = 'output_symbols'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 生成1000个随机包含对号或叉号的图片
for i in range(1000):
    # 随机选择符号
    symbol_char = random.choice(symbols)

    # 生成符号图像
    img = generate_symbol(symbol_char)
    if img is None:
        continue

    # 保存图像，使用唯一编号作为文件名的一部分
    filename = f"symbol_{i + 1}.png"
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    print(f"Saved {filepath}")