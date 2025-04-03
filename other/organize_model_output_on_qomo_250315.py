import os
import shutil

def move_files(src_dir, dest_dir):
    """
    遍历src_dir及其子目录下的所有文件，
    如果dest_dir中不存在同名文件则将文件移动过去，
    否则跳过该文件。
    """
    for root, _, files in os.walk(src_dir):
        for file in files:
            src_file = os.path.join(root, file)
            dest_file = os.path.join(dest_dir, file)
            # 判断目标路径是否存在同名文件
            if not os.path.exists(dest_file):
                print(f"移动文件: {src_file} -> {dest_file}")
                shutil.move(src_file, dest_file)
            else:
                print(f"文件 {dest_file} 已存在，跳过 {src_file}")

if __name__ == '__main__':
    # 设置A、B和新文件夹的路径
    A_path = '/data4/2019swh/b1850_exp'         # 替换为实际的A路径
    B_path = '/data4/2019swh/b1850_exp_process'         # 替换为实际的B路径
    new_path = '/data4/2019swh/model_output_organize' # 替换为实际的新文件夹路径

    # 确保新的文件夹存在
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    print("正在处理A路径下的文件...")
    move_files(A_path, new_path)

    print("正在处理B路径下的文件...")
    move_files(B_path, new_path)