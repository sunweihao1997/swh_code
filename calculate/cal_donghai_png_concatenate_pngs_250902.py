from PIL import Image
import os
import math

# 图片文件夹路径
folder = "/mnt/f/wsl_plot/donghai_ship/single_ship"

# 只取 .png，并做个自然顺序排序（可按需去掉）
files = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
files.sort()
images = [Image.open(os.path.join(folder, f)) for f in files]

# 固定总列数为 4 列
cols = 4
n = len(images)
if n == 0:
    raise SystemExit("没有找到 png 图片")

# 用第一张图的尺寸作为基准
img_w, img_h = images[0].size

# 每列的行数（最后一列可能不足）
rows_per_col = math.ceil(n / cols)

# 先把“列”拼出来：每列最多 rows_per_col 张竖着拼
new_imgs = []
for i in range(0, n, rows_per_col):
    col_imgs = images[i:i + rows_per_col]
    total_h = img_h * len(col_imgs)
    new_im = Image.new("RGB", (img_w, total_h), (255, 255, 255))
    for j, im in enumerate(col_imgs):
        new_im.paste(im, (0, j * img_h))
    new_imgs.append(new_im)

# 只保留前 4 列（万一图片很少也没关系）
new_imgs = new_imgs[:cols]

# 横向再把 4 列拼接起来
total_w = img_w * len(new_imgs)
max_h = max(im.height for im in new_imgs)  # 一般都等于 rows_per_col * img_h
final_im = Image.new("RGB", (total_w, max_h), (255, 255, 255))
for k, im in enumerate(new_imgs):
    final_im.paste(im, (k * img_w, 0))

# 保存结果
final_im.save("/mnt/f/wsl_plot/donghai_ship/All_ships_scatter_4vars.png")
print("Done.")

