import os
import shutil

def classify_files(base_dir):
    """
    遍历 base_dir 下的所有文件（不再递归子文件夹，因为假设所有文件已经集中在 base_dir 下），
    根据文件名第一个'.'前面的部分作为前缀创建子文件夹，
    并将同一前缀的文件移动到对应子文件夹内。
    """
    # 列出 base_dir 下所有文件（排除文件夹）
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isfile(item_path):
            # 使用文件名中第一个 '.' 之前的部分作为分类依据
            prefix = item.split('.')[0]
            dest_folder = os.path.join(base_dir, prefix)
            # 如果目标子文件夹不存在，则创建它
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)
            # 构造目标文件路径
            dest_path = os.path.join(dest_folder, item)
            print(f"将文件 {item_path} 移动到 {dest_path}")
            shutil.move(item_path, dest_path)

if __name__ == '__main__':
    # 设置存有所有文件的新路径
    new_path = '/data4/2019swh/model_output_organize'  # 替换为你的实际路径
    classify_files(new_path)