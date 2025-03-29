import os
import shutil


def clear_folder(folder_path):
    """
    清空指定文件夹下的所有文件和子文件夹。

    :param folder_path: 要清空的文件夹路径
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"文件夹 {folder_path} 不存在！")
        return

    # 遍历文件夹中的所有内容
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)  # 获取完整路径
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                # 如果是文件或符号链接，直接删除
                os.unlink(item_path)
                print(f"已删除文件: {item_path}")
            elif os.path.isdir(item_path):
                # 如果是子文件夹，递归删除
                shutil.rmtree(item_path)
                print(f"已删除文件夹: {item_path}")
        except Exception as e:
            print(f"无法删除 {item_path}: {e}")


# 示例用法
folder_to_clear = r"D:\Program\Graduation Project\user_data\example_folder"
clear_folder(folder_to_clear)