import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pypinyin

# 文件路径
file_path = "/mnt/f/数据_东海局/salinity/盐度.xlsx"

# 读取所有sheet（除第一个）
xls = pd.ExcelFile(file_path)
station_sheets = xls.sheet_names[1:]

station_data = {}

# 读取数据并将sheet名转为拼音
for sheet in station_sheets:
    df = pd.read_excel(file_path, sheet_name=sheet)

    # 假设第1列是时间，第2列是盐度
    time_col = df.columns[0]
    value_col = df.columns[1]

    df = df[[time_col, value_col]]
    df.columns = ['time', sheet]
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    # sheet名转拼音
    pinyin_name = "_".join(pypinyin.lazy_pinyin(sheet))
    station_data[pinyin_name] = df.rename(columns={sheet: pinyin_name})

# 合并数据（只保留所有站点都有值的时间）
merged = pd.concat(station_data.values(), axis=1, join='inner').sort_index()

# 计算相关系数
corr_matrix = merged.corr()

# 绘图
plt.figure(figsize=(8, 6))
ax = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True,
                 cbar_kws={"shrink": .75})

# 移动x轴标签到上方
ax.xaxis.set_label_position('top')
ax.xaxis.tick_top()

# 倾斜x轴标签
plt.xticks(rotation=45, ha='left')
plt.yticks(rotation=0)

plt.tight_layout()
plt.show()
