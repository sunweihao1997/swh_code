# -*- coding: utf-8 -*-
"""
Sheshan 2022-09-01 ~ 2022-12-31
从盐度计算氯度并作图（12 点平滑、刻度美化、标注极大值）。
氯度换算：氯度(mg/L) = 盐度 / 1.80655 * 1000
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ========= 配置区 =========
DATA_PATH    = Path("/mnt/f/数据_东海局/salinity/sheshan_new.xlsx")
SHEET_NAME   = 0   # 佘山文件只有一个表，直接用索引 0
DATE_COL = "时间"
SAL_COL  = "盐度"
STATION_NAME = "sheshan"
STATION_CODE = "06414"
START = "2022-09-01 00:00:00"
END   = "2022-12-31 23:59:59"

# —— 阈值（这里也采用原脚本的 0.45 盐度阈值换算）—— #
SAL_THRESHOLD = 0.45
CL_THRESHOLD  = SAL_THRESHOLD / 1.80655 * 1000  # ≈ 249 mg/L

OUT_PNG = "/mnt/f/wsl_plot/donghai/salinity/06414_sheshan_Cl_20220901-20221231.png"
OUT_PDF = "/mnt/f/wsl_plot/donghai/salinity/06414_sheshan_Cl_20220901-20221231.pdf"
# ========================

# 读取 Excel
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME, header=0)

# 清洗
df.columns = [str(c).strip() for c in df.columns]
df = df.drop_duplicates()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])
df[SAL_COL] = pd.to_numeric(df[SAL_COL], errors="coerce")
df = df.dropna(subset=[SAL_COL])

# 时间窗口
mask = (df[DATE_COL] >= pd.to_datetime(START)) & (df[DATE_COL] <= pd.to_datetime(END))
dfx = df.loc[mask].sort_values(DATE_COL).copy()

# —— 盐度 → 氯度( mg/L ) —— #
dfx["chlorinity_mgL"] = dfx[SAL_COL] / 1.80655 * 1000

# —— 12 点平滑（居中） —— #
dfx["cl_ma12"] = dfx["chlorinity_mgL"].rolling(window=12, center=True, min_periods=1).mean()

# —— 全局极大值（按“氯度原始值”） —— #
idx_max = dfx["chlorinity_mgL"].idxmax()
t_max = dfx.loc[idx_max, DATE_COL]
y_max = dfx.loc[idx_max, "chlorinity_mgL"]

# ===================== 画图 =====================
fig = plt.figure(figsize=(14, 5.2), dpi=140)
ax = plt.gca()

# 原始氯度曲线
markevery = max(len(dfx) // 220, 1)
ax.plot(
    dfx[DATE_COL], dfx["chlorinity_mgL"],
    linewidth=1.2, alpha=0.85,
    marker=".", markersize=2, markevery=markevery,
    label="Chlorinity (raw)"
)

# 12 点平滑曲线
ax.plot(
    dfx[DATE_COL], dfx["cl_ma12"],
    linewidth=1.9, alpha=0.95,
    label="MA(12)"
)

# 阈值线（由 0.45 盐度换算而来）
ax.axhline(y=CL_THRESHOLD, color="red", linestyle="--", linewidth=1.0,
           label=f"threshold ≈ {CL_THRESHOLD:.0f} mg/L (S=0.45)")

# —— x 轴刻度：确保显示“2022-9” —— #
ax.xaxis.set_major_locator(mdates.MonthLocator())
def month_formatter(x, pos):
    d = mdates.num2date(x)
    return f"{d.year}-{d.month}"   # 不补零，如 2022-9
ax.xaxis.set_major_formatter(FuncFormatter(month_formatter))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))

# 刻度与网格美化
ax.tick_params(axis="x", which="major", length=6, width=1.0, pad=6)
ax.tick_params(axis="x", which="minor", length=3, width=0.8)
ax.tick_params(axis="y", which="both", length=4, width=1.0)

ax.grid(True, which="major", linestyle="--", alpha=0.28)
ax.grid(True, which="minor", linestyle=":",  alpha=0.18)

# 标注全局极大值
ax.scatter([t_max], [y_max], s=42, zorder=5)
ax.annotate(
    f"MAX = {y_max:.0f} mg/L\n{t_max:%Y-%m-%d %H:%M}",
    xy=(t_max, y_max),
    xytext=(0.52, 0.86), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", lw=1.0),
    ha="left", va="top", fontsize=9,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6", alpha=0.95)
)

# 其它外观
plt.margins(x=0.01)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

title_station = f"{STATION_NAME}({STATION_CODE})" if STATION_CODE else STATION_NAME
ax.set_title("SheShan 2022.9.1–12.31 Chlorinity Time Series", fontsize=15, pad=10)
ax.set_xlabel("Time (months shown as YYYY-M)")
ax.set_ylabel("Chlorinity (mg/L)")

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.savefig(OUT_PDF, dpi=500)
# plt.show()
