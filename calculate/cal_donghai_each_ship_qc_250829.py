'''
2025-8-29
This script is to get the ship IDs from the Donghai ship data
'''
import pandas as pd
import os   
import numpy as np

ship_data_path = "/mnt/f/ERA5_ship/donghai_ship_ERA5_QC/"

ship_list = os.listdir(ship_data_path)

ship_list_simple = [] ; nan_count_p = np.array([]) ; bad_qc_count_p = np.array([])
for ship in ship_list:
    print(f"Processing ship file: {ship} ...")

    single_ship = pd.read_csv(ship_data_path + ship)

    nan_count = 0 ; bad_qc_count = 0
    total_length = len(single_ship)

    # 去掉前缀
    if ship.startswith("202506_"):
        ship = ship[len("202506_"):]
    # 去掉后缀
    if ship.endswith(".csv"):
        ship = ship[:-len(".csv")]
    ship_list_simple.append(ship)

    # ---- First. Count how many NAN records ----

    for index, row in single_ship.iterrows():
        #print(row[1:])
        if any(pd.isna(value) for value in row):
            nan_count += 1

    nan_count_p = np.append(nan_count_p, nan_count/total_length*100)
    
    # ---- Second. Count how many qc_overall != 1 records ----
    single_ship = single_ship.dropna()
    for index, row in single_ship.iterrows():
        if row["qc_overall"] != 1:
            bad_qc_count += 1

    bad_qc_count_p = np.append(bad_qc_count_p, bad_qc_count/total_length*100)

    print(f"Ship ID: {ship}, Total records: {total_length}, NAN records: {nan_count}, Bad QC records: {bad_qc_count}")

# ============= Plotting =============
import matplotlib.pyplot as plt
    
fig, ax = plt.subplots(figsize=(8,6))

x = np.arange(len(ship_list_simple))  # 横坐标位置
width = 0.6                # 柱子宽度

fig, ax = plt.subplots(figsize=(20,6))

# 堆叠柱状图
bars1 = ax.bar(x, nan_count_p, width, label='missing ratio')
bars2 = ax.bar(x, bad_qc_count_p, width, bottom=nan_count_p, label='bad ratio')

# 设置标题和坐标轴
ax.set_ylabel('Ratio (%)')
ax.set_xlabel('Ship ID')
#ax.set_title('缺测与未过质控占比')
ax.set_xticks(x)
ax.set_xticklabels(ship_list_simple, rotation=45, ha='right')
ax.legend()


# 改进版：支持百分比格式、最小显示阈值、放在段中心/柱顶，自动识别 0~1 / 0~100
def autolabel(bars, values, bottom_values=None, *,
              percent=True,          # True: 用百分比显示
              min_display=5.0,       # 小于该阈值(%)不显示，避免密集“乱码”
              place='center',        # 'center' 段内居中；'top' 放在段顶
              pad=3,                 # 顶部标注像素偏移
              fmt="{:.1f}%"):        # 文本格式
    vals = np.asarray(values, dtype=float)
    # 处理 bottom
    if bottom_values is None:
        bottom = np.zeros_like(vals)
    else:
        bottom = np.asarray(bottom_values, dtype=float)

    # 自动把 0~1 转为 0~100
    scale = 100.0 if percent and max(vals.max(initial=0), bottom.max(initial=0)) <= 1.0 else 1.0
    vals   = vals   * scale
    bottom = bottom * scale

    for i, bar in enumerate(bars):
        v = float(vals[i])
        if v < min_display:  # 过滤太小的段
            continue
        b = float(bottom[i])

        if place == 'center':
            y  = b + v/2.0
            va = 'center'
            dy = 0
        else:  # 'top'
            y  = b + v
            va = 'bottom'
            dy = pad

        ax.annotate(fmt.format(v) if percent else f"{v:.1f}",
                    xy=(bar.get_x() + bar.get_width()/2.0, y),
                    xytext=(0, dy),
                    textcoords="offset points",
                    ha='center', va=va, fontsize=8)


autolabel(bars1, nan_count_p)
autolabel(bars2, bad_qc_count_p, bottom_values=nan_count_p)

plt.tight_layout()
plt.savefig("/mnt/f/wsl_plot/donghai_ship_missing_badqc_ratio.png", dpi=500)