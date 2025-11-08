# -*- coding: utf-8 -*-
"""
功能：
在 daily（日平均）结果基础上绘制时间序列图：
  - 水位：折线图
  - 流量：柱状图
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib
import os
matplotlib.use('Agg')

# =============================================================================

# 可能存在的字体路径（按需保留/添加）
font_paths = [
    '/mnt/c/Windows/Fonts/msyh.ttf',    # 微软雅黑
    '/mnt/c/Windows/Fonts/simhei.ttf',  # 黑体
    '/mnt/c/Windows/Fonts/msyhbd.ttf',  # 微软雅黑 Bold
]

# 动态注册存在的字体
for p in font_paths:
    if os.path.exists(p):
        font_manager.fontManager.addfont(p)

# 清空缓存并让 Matplotlib 识别新字体（第一次用时）
try:
    font_manager._load_fontmanager(try_read_cache=False)
except Exception:
    pass

# 设置优先字体（会按顺序回退）
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

# =============================================================================

# 读取区间并标准化列名
def read_intervals_simple(path, sheet_name=0):
    """
    从 Excel 文件读取时间区间，返回包含列：
    start, end, label, 站名（如果存在）
    """
    df = pd.read_excel(path, sheet_name=sheet_name) # sheet_name=0 表示第一个工作表
    print(df.columns.tolist())

    start_col = "开始日期" ; end_col = "结束日期"


    # 基础转换
    out = pd.DataFrame({
        "start": pd.to_datetime(df[start_col], errors="coerce"),
        "end": pd.to_datetime(df[end_col], errors="coerce"),
    })

    # 丢弃无效行
    out = out.dropna(subset=["start", "end"])
    out = out.query("end >= start").reset_index(drop=True)
    return out


# =============================================================

import pandas as pd
import numpy as np

def fill_gaps_per_station(
    daily: pd.DataFrame,
    cols=("流量日平均", "水位日平均"),
    max_gap_days=7,            # 只插不超过 N 天的连续缺口
    do_ffill_bfill=True        # 插值后再做一次前后向回填（可按需关掉）
):
    out = []
    for name, g in daily.groupby("站名", as_index=False):
        g = g.sort_values("日期").copy()

        # 1) 建立完整逐日索引（从该站最小到最大日期）
        full_idx = pd.date_range(g["日期"].min(), g["日期"].max(), freq="D")
        g = g.set_index("日期").reindex(full_idx)
        g.index.name = "日期"
        g["站名"] = name  # 补回站名

        # 2) 对每列做“限长时间插值”
        for col in cols:
            # 仅插内部缺口，limit 限制最大连续 NaN 天数
            g[col] = g[col].interpolate(method="time", limit=max_gap_days, limit_area="inside")

        # 3) 可选：对仍残留的 NaN（多在两端或超长缺口），做一次 ffill/bfill
        if do_ffill_bfill:
            for col in cols:
                g[col] = g[col].ffill().bfill()

        out.append(g.reset_index())

    return pd.concat(out, ignore_index=True)



# 假设 daily 已经计算好
# 如果要读取上一步的结果，可以取消下一行注释：
daily = pd.read_excel("/mnt/f/data_donghai/salinity/大通径流量_日平均.xlsx", sheet_name="日平均")

daily_roll = fill_with_rolling_mean(daily)
daily_filled = fill_gaps_per_station(
    daily,
    cols=("流量日平均", "水位日平均"),
    max_gap_days=10,
    do_ffill_bfill=True
)



# Read Intrusion Process
date_intrusion = "/mnt/f/data_donghai/salinity/咸潮入侵过程统计_优化版.xlsx"
intrusion_intervals = read_intervals_simple(date_intrusion, sheet_name=0)
print(intrusion_intervals)

daily_filled["日期"] = pd.to_datetime(daily_filled["日期"]).dt.normalize()
intrusion_intervals["start"] = pd.to_datetime(intrusion_intervals["start"]).dt.normalize()
intrusion_intervals["end"]   = pd.to_datetime(intrusion_intervals["end"]).dt.normalize()

# === 限定时间范围 ===
start_date = pd.Timestamp("2022-08-31")
end_date   = pd.Timestamp("2022-12-31")
mask = (daily_filled["日期"] >= start_date) & (daily_filled["日期"] <= end_date)
daily_filled = daily_filled.loc[mask].copy()

# 按站名分别绘图
for name, df_site in daily_filled.groupby("站名"):
    fig, ax1 = plt.subplots(figsize=(20, 5))
    plt.title(f"Datong Station Daily Mean Water Level and Runoff", fontsize=16)
    
    # 左坐标轴（流量）
    ax1.bar(
        df_site["日期"],
        df_site["流量日平均"],
        width=pd.Timedelta(days=1),   # 一天宽度
        align="edge",                 # 左对齐，代表“覆盖这一天”
        color="lightblue",
        label="runoff",
        alpha=0.75,
    )
    ax1.set_ylabel("daily runoff (m³/s)", color="blue")
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # 右坐标轴（水位）
    ax2 = ax1.twinx()
    ax2.plot(df_site["日期"], df_site["水位日平均"], color="red", linewidth=2, label="water level")
    ax2.set_ylabel("water level (m)", color="red")
    ax2.tick_params(axis='y', labelcolor='red')

    # 合并两轴图例（关键修复）
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")

    # === add intrusion intervals ===
    first = True
    tmin, tmax = df_site["日期"].min(), df_site["日期"].max()
    for _, row in intrusion_intervals.iterrows():
        s = max(row["start"], tmin)
        e = min(row["end"],   tmax)
        if e >= s:
            ax1.axvspan(
                s, e + pd.Timedelta(days=1),   # 让“结束日”也被涂色
                color="#FFA07A", alpha=0.2,
                label="咸潮入侵期" if first else None
            )
            first = False

    # 美化
    plt.xlabel("Date")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # 保存每个站的图像
    plt.savefig(f"/mnt/f/wsl_plot/donghai/salinity/{name}_日平均水位流量图_v2.png", dpi=300)
    plt.close()

print("✅ 图像已保存为：<站名>_日平均水位流量图.png")
