# -*- coding: utf-8 -*-
"""
QingCaoSha 2022-09-05 ~ 2022-12-12 氯离子(或盐度)时间序列
改进点：
1) x 轴主刻度显示到月份，格式如 2022-9（包含 2022-9）
2) 刻度/网格美化
3) 6 点平滑曲线
4) 标注全局极大值
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import numpy as np

# ========= 配置区(按你的实际情况改) =========
DATA_PATH = Path("/mnt/f/数据_东海局/salinity/qingcaosha_new.xlsx")  # ← 改成你的文件
SHEET_NAME = 0
STATION_NAME = "QingCaoSha"
STATION_CODE = ""              # 若无可留空
DATE_COL = "时间"
SAL_COL = "数值"               # 文件中的数值列名
START = "2022-08-15 00:00:00"
END   = "2022-12-31 23:59:59"
THRESHOLD = 250                # 参考阈值（原来画的红虚线）
OUT_PNG = "/mnt/f/wsl_plot/donghai/salinity/QingcaoSha_Chloride_20220815-20221212.png"
OUT_PDF = "/mnt/f/wsl_plot/donghai/salinity/QingcaoSha_Chloride_20220815-20221212.pdf"
# =========================================

# 1) 读数据(自动兼容 Excel / CSV)
suffix = DATA_PATH.suffix.lower()
if suffix in [".xlsx", ".xls"]:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=0)
    # 如果没有读到“时间/数值”两列，则尝试用第2行做表头
    if not ({DATE_COL, SAL_COL} <= set([str(c).strip() for c in df.columns])):
        df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=1)
elif suffix in [".csv", ".txt"]:
    df = pd.read_csv(DATA_PATH)
else:
    raise ValueError("不支持的文件类型，请提供 .xlsx/.xls/.csv")

# 2) 清洗
df.columns = [str(c).strip() for c in df.columns]
df = df.drop_duplicates()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])
df[SAL_COL] = pd.to_numeric(df[SAL_COL], errors="coerce")

# 3) 时间窗口
mask = (df[DATE_COL] >= pd.to_datetime(START)) & (df[DATE_COL] <= pd.to_datetime(END))
df = df.loc[mask].sort_values(DATE_COL)
df = df.dropna(subset=[SAL_COL])

# 4) 6 点平滑（居中，缺失时也能算）
df["smooth6"] = df[SAL_COL].rolling(window=24, center=True, min_periods=1).mean()

# 5) 计算极大值（全局最大）
imax = int(df[SAL_COL].idxmax())
t_max = df.loc[imax, DATE_COL]
y_max = df.loc[imax, SAL_COL]

# ===================== 画图 =====================
fig = plt.figure(figsize=(20, 8), dpi=500)
ax = plt.gca()

# 原始曲线：连线 + 适度抽样标记点
markevery = max(len(df) // 220, 1)
ax.plot(df[DATE_COL], df[SAL_COL],
        linewidth=1.2, alpha=0.85,
        marker=".", markersize=2, markevery=markevery, color="red",
        label="Hourly")

# 6 点平滑曲线
ax.plot(df[DATE_COL], df["smooth6"],
        linewidth=1.8, alpha=0.95, color="black",
        label="Daily Smoothed")

# 阈值线
ax.axhline(y=THRESHOLD, color="red", linestyle="--",
           linewidth=1.5, label=f"y={THRESHOLD}")

# —— x 轴刻度与格式：确保显示“2022-9” —— #
# 主刻度：每月
ax.xaxis.set_major_locator(mdates.MonthLocator())
# 自定义格式器：年份-月份(不补零)，如 2022-9
def month_formatter(x, pos):
    d = mdates.num2date(x)
    return f"{d.year}-{d.month}"
ax.xaxis.set_major_formatter(FuncFormatter(month_formatter))

# 次刻度：每周一
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

# 刻度样式更美观
ax.tick_params(axis="x", which="major", length=6, width=1.0, pad=6)
#ax.tick_params(axis="x", which="minor", length=3, width=0.8)
ax.tick_params(axis="y", which="both", length=4, width=1.0)
#fig.autofmt_xdate(rotation=25, ha="right")

# 网格：主网格稍粗，次网格更淡
ax.grid(True, which="major", linestyle="--", alpha=0.28)
ax.grid(True, which="minor", linestyle=":", alpha=0.18)

# 标注全局极大值
ax.scatter([t_max], [y_max], s=25, zorder=5)
ax.annotate(
    f"MAX = {y_max:.0f}\n{t_max:%Y-%m-%d %H:%M}",
    xy=(t_max, y_max),
    xytext=(0.5, 0.85), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=1.0),
    ha="left", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", alpha=0.9)
)

# 其它外观
plt.margins(x=0.01)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

title_station = f"{STATION_NAME}({STATION_CODE})" if STATION_NAME and STATION_CODE else (STATION_NAME or STATION_CODE or "Station")
ax.set_title(f"{title_station} 2022.9.5–12.12 Chloride concentration Time Series", fontsize=15, pad=10)
ax.set_xlabel("Time (months shown as YYYY-M)")  # 明确说明刻度格式
ax.set_ylabel("Chloride concentration")
ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=15)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF, dpi=500)
# plt.show()
