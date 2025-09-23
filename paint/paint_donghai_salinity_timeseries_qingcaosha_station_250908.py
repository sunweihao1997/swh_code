'''
20250908
This script is to plot the salinity timeseries for another station (2022-09-05 ~ 2022-12-12),
keeping the same plotting style as the previous figure.
'''
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ========= 配置区(按你的实际情况改) =========
# 新站点数据文件路径（支持 .xlsx/.xls/.csv）
DATA_PATH = Path("/mnt/f/数据_东海局/salinity/qingcaosha.xlsx")  # ← 改成你的新文件名
SHEET_NAME = 0
STATION_NAME = "QingCaoSha"   # 仅用于标题展示
STATION_CODE = ""             # 若无可留空
DATE_COL = "时间"
SAL_COL = "数值"               # 新文件中为“数值”
START = "2022-09-05 00:00:00"
END   = "2022-12-12 23:59:59"
# =========================================

# 1) 读数据(自动兼容 Excel / CSV；若首行是标题行会自动下移一行)
suffix = DATA_PATH.suffix.lower()
if suffix in [".xlsx", ".xls"]:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=0)
    # 如果没有读到“时间/数值”两列，则尝试用第2行做表头
    if not ({DATE_COL, SAL_COL} <= set([str(c).strip() for c in df.columns])):
        df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=1)
elif suffix in [".csv", ".txt"]:
    df = pd.read_csv(DATA_PATH)
else:
    raise ValueError("不支持的文件类型，请提供 .xlsx / .csv")

# 2) 基本清洗：统一列名空白、去重、时间解析
df.columns = [str(c).strip() for c in df.columns]
df = df.drop_duplicates()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])

# 3) 不做站点字段过滤（该文件即为单站数据）
dfx = df.copy()

# 4) 时间窗口过滤
mask = (dfx[DATE_COL] >= pd.to_datetime(START)) & (dfx[DATE_COL] <= pd.to_datetime(END))
dfx = dfx.loc[mask].sort_values(DATE_COL)

# 5) 数值列转数值并去掉缺失
dfx[SAL_COL] = pd.to_numeric(dfx[SAL_COL], errors="coerce")
dfx = dfx.dropna(subset=[SAL_COL])

# ========== 画图(保持与你上个脚本一致的风格) ==========
fig = plt.figure(figsize=(12, 5), dpi=140)
ax = plt.gca()

# 原始数据：连线 + 适度抽样的标记点
markevery = max(len(dfx) // 200, 1)
ax.plot(
    dfx[DATE_COL], dfx[SAL_COL],
    linewidth=1.2, alpha=0.85,
    marker=".", markersize=2, markevery=markevery,
    label="Salinity"
)

ax.axhline(y=250, color="red", linestyle="--", linewidth=1.0, label="y=0.45")

# 网格与边距
ax.grid(True, which="both", linestyle="--", alpha=0.25)
plt.margins(x=0.01)

# 轴与刻度：月份主刻度 + 每周次刻度
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
fig.autofmt_xdate()

# 标签与标题
title_station = f"{STATION_NAME}({STATION_CODE})" if STATION_NAME and STATION_CODE else (STATION_NAME or STATION_CODE or "Station")
ax.set_title(f"{title_station} 2022.9.5–12.12 Chloride concentration Time Series", fontsize=14, pad=10)
ax.set_xlabel("Time")
ax.set_ylabel("Chloride concentration")
ax.legend(frameon=False)

# 边框微调
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

# ====== 导出路径（按需修改）======
plt.savefig('/mnt/f/wsl_plot/donghai/salinity/QingcaoSha_Chloride concentration_20220905-20221212.pdf', dpi=500)
# plt.show()
