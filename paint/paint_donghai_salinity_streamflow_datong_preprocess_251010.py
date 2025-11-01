'''
2025-10-10
This script is preprocess for ploting stramflow time series for Datong station
'''
import pandas as pd

# 1) 读文件：如果你的表没有表头，用 header=None；如果第一列表头就是 year，把 header=None 去掉或改成 header=0
df = pd.read_excel("/mnt/f/数据_东海局/salinity/大通径流量-历史多年.xlsx", header=None)

# 2) 取出最左侧一列作为年份列
year_col = df.iloc[:, 0]

# 3) 只保留 2000–2021 的年份
mask = year_col.between(2000, 2021)
df = df[mask]

# 4) 按年份分组计数
counts = df.groupby(df.iloc[:, 0]).size().sort_index()

# 5) 打印与保存
print(counts)
counts.to_csv("year_counts.csv", header=["count"])
