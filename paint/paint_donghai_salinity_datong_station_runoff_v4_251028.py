# -*- coding: utf-8 -*-
"""
功能：
在 daily（日平均）结果基础上绘制时间序列图：
  - 左轴：流量（柱）
  - 右轴：水位（线）
  - 按 CSV 里的区间用 axvspan 着色
"""

import matplotlib
matplotlib.use('Agg')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.dates as mdates

# =============================================================================
# 字体（按需存在则注册）
font_paths = [
    '/mnt/c/Windows/Fonts/msyh.ttf',
    '/mnt/c/Windows/Fonts/simhei.ttf',
    '/mnt/c/Windows/Fonts/msyhbd.ttf',
]
for p in font_paths:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)
try:
    font_manager._load_fontmanager(try_read_cache=False)
except Exception:
    pass

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False
# =============================================================================

def fill_with_rolling_mean(daily: pd.DataFrame, cols=("流量日平均", "水位日平均"), window=7, min_periods=3):
    out = []
    for name, g in daily.groupby("站名", as_index=False):
        g = g.sort_values("日期").copy()
        for col in cols:
            roll = g[col].rolling(window=window, min_periods=min_periods, center=True).mean()
            g[col] = g[col].fillna(roll)
        out.append(g)
    return pd.concat(out, ignore_index=True)

# —— 新增：从 CSV 读取 interval（最小化假设：列名是 start/end 或 开始日期/结束日期）——
def read_intervals_csv_simple(path: str):
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}  # 不区分大小写匹配

    # 支持的列名同义词
    start_candidates = ["start", "start_date", "开始日期", "开始时间", "起始", "起始日期"]
    end_candidates   = ["end", "end_date", "结束日期", "结束时间", "终止", "终止日期"]

    def pick(cands):
        for k in cands:
            if k in cols:
                return cols[k]
        return None

    start_col = pick(start_candidates)
    end_col   = pick(end_candidates)
    if start_col is None or end_col is None:
        raise ValueError(f"CSV 中未找到开始/结束列。实际列名：{list(df.columns)}")

    out = pd.DataFrame({
        "start": pd.to_datetime(df[start_col], errors="coerce").dt.normalize(),
        "end":   pd.to_datetime(df[end_col],   errors="coerce").dt.normalize(),
    })
    out = out.dropna(subset=["start", "end"])
    out = out.query("end >= start").reset_index(drop=True)
    return out


# —— 新增：合并重叠/相邻区间，输出 [(s,e_plot), ...] 供 axvspan 使用 ——
def merge_intervals(df_intervals: pd.DataFrame,
                    tmin: pd.Timestamp = None,
                    tmax: pd.Timestamp = None,
                    inclusive_end: bool = True,
                    join_adjacent: bool = True):
    iv = df_intervals.copy()
    iv["start"] = pd.to_datetime(iv["start"]).dt.normalize()
    iv["end"]   = pd.to_datetime(iv["end"]).dt.normalize()

    # 裁剪到目标时间窗
    if tmin is not None:
        iv["start"] = iv["start"].clip(lower=tmin)
    if tmax is not None:
        iv["end"]   = iv["end"].clip(upper=tmax)

    iv = iv.dropna(subset=["start", "end"])
    iv = iv[iv["end"] >= iv["start"]].sort_values("start").reset_index(drop=True)

    merged = []
    for _, row in iv.iterrows():
        s, e = row["start"], row["end"]
        if not merged:
            merged.append([s, e])
            continue
        s0, e0 = merged[-1]
        tol = pd.Timedelta(days=1) if (inclusive_end and join_adjacent) else pd.Timedelta(0)
        if s <= e0 + tol:                 # 重叠或相邻 (含结束日)
            merged[-1][1] = max(e0, e)
        else:
            merged.append([s, e])

    shade = []
    for s, e in merged:
        e_plot = e + pd.Timedelta(days=1) if inclusive_end else e  # axvspan 右开
        shade.append((s, e_plot))
    return shade

# =============================================================
def fill_gaps_per_station(
    daily: pd.DataFrame,
    cols=("流量日平均", "水位日平均"),
    max_gap_days=7,
    do_ffill_bfill=True
):
    out = []
    for name, g in daily.groupby("站名", as_index=False):
        g = g.sort_values("日期").copy()
        full_idx = pd.date_range(g["日期"].min(), g["日期"].max(), freq="D")
        g = g.set_index("日期").reindex(full_idx)
        g.index.name = "日期"
        g["站名"] = name
        for col in cols:
            g[col] = g[col].interpolate(method="time", limit=max_gap_days, limit_area="inside")
        if do_ffill_bfill:
            for col in cols:
                g[col] = g[col].ffill().bfill()
        out.append(g.reset_index())
    return pd.concat(out, ignore_index=True)

# ================= 主流程 =================
# 1) 日平均数据
daily = pd.read_excel("/mnt/f/data_donghai/salinity/大通径流量_日平均.xlsx", sheet_name="日平均")

daily_filled = fill_gaps_per_station(
    daily, cols=("流量日平均", "水位日平均"),
    max_gap_days=10, do_ffill_bfill=True
)
daily_filled["日期"] = pd.to_datetime(daily_filled["日期"]).dt.normalize()

# 2) 读取 CSV 区间
csv_path = "/mnt/f/wsl_plot/donghai/salinity/baogang_daily_series/ALL_sheets_exceed_intervals.csv"
intervals = read_intervals_csv_simple(csv_path)

# 3) 限定绘图时间窗（如需）
start_date = pd.Timestamp("2024-09-15")
end_date   = pd.Timestamp("2025-04-15")
mask = (daily_filled["日期"] >= start_date) & (daily_filled["日期"] <= end_date)
daily_filled = daily_filled.loc[mask].copy()

# 4) 绘图
out_dir = "/mnt/f/wsl_plot/donghai/salinity"
os.makedirs(out_dir, exist_ok=True)

for name, df_site in daily_filled.groupby("站名"):
    df_site = df_site.sort_values("日期").copy()
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.set_title("Datong Station Daily Mean Water Level and Runoff", fontsize=16)

    # 左轴：流量（柱，左对齐+宽度1天）
    ax1.bar(
        df_site["日期"], df_site["流量日平均"],
        width=pd.Timedelta(days=1), align="edge",
        color="lightblue", alpha=0.75, label="runoff"
    )
    ax1.set_ylabel("daily runoff (m^3/s)", color="blue")
    ax1.tick_params(axis='y', labelcolor='blue')

    # 右轴：水位（线）
    ax2 = ax1.twinx()
    ax2.plot(df_site["日期"], df_site["水位日平均"], color="red", linewidth=2, label="water level")
    ax2.set_ylabel("water level (m)", color="red")
    ax2.tick_params(axis='y', labelcolor='red')

    # —— 生成 shade_intervals（先按站点时间窗裁剪并合并）——
    tmin, tmax = df_site["日期"].min(), df_site["日期"].max()
    shade_intervals = merge_intervals(intervals, tmin=tmin, tmax=tmax,
                                      inclusive_end=True, join_adjacent=True)

    # —— 参照你的 quiver 示例：循环画 axvspan —— 
    for (s, e) in shade_intervals:
        ax1.axvspan(s, e, alpha=0.2, color="#FFA07A")

    # 图例与坐标格式
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ax1.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()

    ax1.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, f"/mnt/f/wsl_plot/donghai/salinity/{name}_日平均水位流量图_v4.png"), dpi=300)
    plt.close()

print("✅ 图像已保存：<站名>_日平均水位流量_含CSV区间.png")
