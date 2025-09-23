'''
20250908
Plot salinity timeseries for the Chongming sheet (title uses 'chongming').
Style matches your previous plots (no smoothing).
'''
# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ========= 配置区 =========
DATA_PATH    = Path("/mnt/f/数据_东海局/salinity/wuhaogou_chongming.xlsx")  # ← 改成实际路径
SHEET_HINTS  = ["05454崇明", "崇明", "05454"]  # 按顺序尝试匹配的 sheet 名
DATE_COL = "日期"
SAL_COL  = "盐度"
STATION_NAME = "chongming"   # 只用于标题（英文）
STATION_CODE = "05454"
START = "2022-09-01 00:00:00"
END   = "2022-12-31 23:59:59"
OUTFIG = "/mnt/f/wsl_plot/donghai/salinity/05454_chongming_20220905-20221212.png"
# ========================

# 读取指定 sheet（自动选择引擎）
suffix = DATA_PATH.suffix.lower()
engine = "openpyxl" if suffix == ".xlsx" else ("xlrd" if suffix == ".xls" else None)
if engine is None:
    raise ValueError("不支持的文件类型，请提供 .xlsx / .xls")

xf = pd.ExcelFile(DATA_PATH, engine=engine)

# 选择包含“崇明/05454”的 sheet
target_sheet = None
for hint in SHEET_HINTS:
    if hint in xf.sheet_names:
        target_sheet = hint
        break
if target_sheet is None:
    # 模糊匹配
    cand = [s for s in xf.sheet_names if ("崇明" in s or "05454" in s)]
    target_sheet = cand[0] if cand else xf.sheet_names[0]

df = pd.read_excel(xf, sheet_name=target_sheet, header=0)

# 清洗
df.columns = [str(c).strip() for c in df.columns]
df = df.drop_duplicates()
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.dropna(subset=[DATE_COL])
df[SAL_COL] = pd.to_numeric(df[SAL_COL], errors="coerce")
df = df.dropna(subset=[SAL_COL])

# 时间窗口
mask = (df[DATE_COL] >= pd.to_datetime(START)) & (df[DATE_COL] <= pd.to_datetime(END))
dfx = df.loc[mask].sort_values(DATE_COL)

# 画图（保持风格一致）
fig = plt.figure(figsize=(12, 5), dpi=140)
ax = plt.gca()

markevery = max(len(dfx) // 200, 1)
ax.plot(
    dfx[DATE_COL], dfx[SAL_COL],
    linewidth=1.2, alpha=0.85,
    marker=".", markersize=2, markevery=markevery,
    label="Salinity"
)
ax.axhline(y=0.45, color="black", linestyle="--", linewidth=1.0, label="y=0.45")

ax.grid(True, which="both", linestyle="--", alpha=0.25)
plt.margins(x=0.01)

ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
fig.autofmt_xdate()

title_station = f"{STATION_NAME}({STATION_CODE})" if STATION_CODE else STATION_NAME
ax.set_title(f"ChongMing 2022.9.1–12.31 Salinity Time Series", fontsize=14, pad=10)
ax.set_xlabel("Time")
ax.set_ylabel("Salinity")
ax.legend(frameon=False)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig(OUTFIG, dpi=500)
# plt.show()
