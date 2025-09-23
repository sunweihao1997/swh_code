'''
20250908
This script is to plot the salinity timeseries at Sheshan station in Donghai.
'''
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os

# ========= 配置区(按你的实际情况改) =========
# 数据文件路径：既支持 .xlsx 也支持 .csv
DATA_PATH = Path("/mnt/f/数据_东海局/salinity/sheshan_new.xlsx")   # 改成你的文件名
SHEET_NAME = 0         # 如果是 Excel 且有多表，可改成表名
STATION_NAME = "SheShan"   # 或者用 STATION_CODE = "06414"
STATION_CODE = "06414" # 二选一或都保留
DATE_COL = "时间"
SAL_COL = "盐度"
START = "2022-09-01"
END   = "2022-12-31 23:59:59"
# =========================================

# 1) 读数据(自动兼容 Excel / CSV)
#print(os.listdir('/mnt/f/数据_东海局/salinity/'))
suffix = DATA_PATH.suffix.lower()
if suffix in [".xlsx", ".xls"]:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
elif suffix in [".csv", ".txt"]:
    # 如果 CSV 为 GBK/ANSI，可加 encoding="gbk"
    df = pd.read_csv(DATA_PATH)
else:
    raise ValueError("不支持的文件类型，请提供 .xlsx / .csv")


# 2) 基本清洗：统一列名空白、去重、时间解析
df.columns = [c.strip() for c in df.columns]
df = df.drop_duplicates()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])


## 3) 站点过滤(尽量稳妥：优先用站代号，其次用站点名)
#if "站代号" in df.columns:
#    dfx = df[df["站代号"].astype(str).str.strip() == STATION_CODE]
#elif "站点名" in df.columns:
#    dfx = df[df["站点名"].astype(str).str.contains(STATION_NAME, na=False)]
#else:
#    dfx = df.copy()  # 若没有这两列，就不过滤

dfx = df
# 4) 时间窗口过滤
mask = (dfx[DATE_COL] >= pd.to_datetime(START)) & (dfx[DATE_COL] <= pd.to_datetime(END))
dfx = dfx.loc[mask].sort_values(DATE_COL)

dfx = dfx.sort_values(DATE_COL)
print(dfx[DATE_COL])

# ========== 画图(美化) ==========
# 中文字体 & 负号
#plt.rcParams["axes.unicode_minus"] = False
#for font in ["Microsoft YaHei", "SimHei", "Heiti TC", "WenQuanYi Zen Hei"]:
#    try:
#        plt.rcParams["font.sans-serif"] = [font]
#        break
#    except Exception:
#        pass

fig = plt.figure(figsize=(12, 5), dpi=140)
ax = plt.gca()

# 原始数据：连线 + 适度抽样的标记点，避免点数过多显得密集
markevery = max(len(dfx) // 200, 1)  # 每隔多少个点画一个标记

ax.plot(
    dfx[DATE_COL], dfx[SAL_COL],
    linewidth=1.2, alpha=0.85,
    marker=".", markersize=2, markevery=markevery,
    label="Salinity"
)

# 网格与边距
ax.grid(True, which="both", linestyle="--", alpha=0.25)
plt.margins(x=0.01)

# 轴与刻度：月份主刻度 + 每周次刻度
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
fig.autofmt_xdate()

# 标签与标题
title_station = f"{STATION_NAME}({STATION_CODE})" if STATION_NAME and STATION_CODE else (STATION_NAME or STATION_CODE or "")
ax.set_title(f"{title_station} 2022.9-12 Salinity Time Series", fontsize=14, pad=10)
ax.set_xlabel("Time")
ax.set_ylabel("Salinity")
ax.legend(frameon=False)

# 边框微调(去掉上右边框)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig('/mnt/f/wsl_plot/donghai/salinity/sheshan_station_salinity_20220901-20221231.png', dpi=500)