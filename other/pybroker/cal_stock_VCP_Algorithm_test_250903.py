'''
20250903
This script is to test cal_stock_VCP_Algorithm_250902
'''
# test_vcp_on_one_stock.py
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt

# ====== 引入你已有的函数（直接复制到同目录的脚本里/或用 from your_module import *） ======
from typing import List, Tuple, Optional, Union, Dict
# 这里假设你已把下列函数放在同一个脚本里：find_swings, extract_contractions_v2, is_valid_vcp_sequence_v2
# 如果它们在你贴的那个脚本中（比如 cal_stock_ATR_shrinking_250901.py），就改成：
from cal_stock_VCP_Algorithm_250902 import find_swings, extract_contractions_v2, is_valid_vcp_sequence_v2

# ====== 1) 拉取数据 ======
symbol = "000001"        # 换成你想测的股票：如 600519、300750...
start_date = "20220101"  # 起始时间
end_date   = "20250902"  # 结束时间（今天/任意日期）
adjust     = "qfq"       # 前复权，更适合技术分析；不复权用 "" 或 "hfq" 后复权

df_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                            start_date=start_date, end_date=end_date, adjust=adjust)

# ====== 2) 清洗为标准OHLCV ======
df = df_raw.rename(columns={
    "日期": "Date", "开盘": "Open", "最高": "High", "最低": "Low", "收盘": "Close", "成交量": "Volume"
}).copy()

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date").sort_index()

for col in ["Open", "High", "Low", "Close", "Volume"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# （可选）把“手”换成股：df["Volume"] = df["Volume"] * 100

# ====== 3) 运行你的VCP收缩检测 ======
# 3.1 摆动点
sw = find_swings(df, window=5)  # window 越大越抗噪，越小越灵敏

# 3.2 提取回撤对（高级版，带抗噪条件）
res = extract_contractions_v2(
    df, sw,
    lookback=300,         # 往回看近90根K
    min_drop=0.05,       # 单段最小回撤 5%
    max_drop=0.35,       # 单段最大回撤 35%
    min_bars_per_leg=1,  # 高低点之间至少隔5根K，防锯齿
    max_last_drop=0.15   # 末端更紧：最后一段 ≤ 15%
)

pairs = res["pairs"]  # [(hi_t, hi_v, lo_t, lo_v, drop), ...]
print(f"\n=== {symbol} 回撤段（近{len(pairs)}段，显示最近5段）===")
for tup in pairs[-5:]:
    hi_t, hi_v, lo_t, lo_v, drop = tup
    print(f"hi@{hi_t.date()}={hi_v:.2f}  →  low@{lo_t.date()}={lo_v:.2f}  drop={drop*100:.2f}%")

# 3.3 检验是否满足VCP收缩序列（最近3段/容差）
chk = is_valid_vcp_sequence_v2(
    res,
    need_n=3,
    tol_drop=0.01,          # 回撤至少逐段缩小 1pct
    enforce_higher_lows=True,
    low_tol=0.0,            # 低点抬高容差（可用 0.02 或 0.1*ATR）
    require_last_tight=True # 强制末段更紧通过
)

print("\n=== 校验结果 ===")
print("valid:", chk["valid"])
print("reason:", chk["reason"])
print("drops (最近N段):", chk["drops"])
print("lows  (最近N段):", chk["lows"])
print("last_drop:", chk["last_drop"])
print("passed_last_tight:", chk["passed_last_tight"])
print("pivot_time:", chk["pivot_time"], "pivot_price:", chk["pivot_price"])

# ====== 4) （可选）可视化最近一段：标注 swing high / low 与 pivot + 配对段 ======
try:
    tail_n = 240  # 画最近240根K
    df_tail = df.tail(tail_n).copy()
    sw_tail = sw.loc[df_tail.index]

    plt.figure(figsize=(11, 5))
    plt.plot(df_tail.index, df_tail["Close"], label="Close", linewidth=1.2)

    # --- 标 swing 高低点（三角点，来自 find_swings）
    hi_mask = ~pd.isna(sw_tail["swing_high"])
    lo_mask = ~pd.isna(sw_tail["swing_low"])
    plt.scatter(
        df_tail.index[hi_mask], sw_tail.loc[hi_mask, "swing_high"],
        marker="^", s=60, label="Swing High"
    )
    plt.scatter(
        df_tail.index[lo_mask], sw_tail.loc[lo_mask, "swing_low"],
        marker="v", s=60, label="Swing Low"
    )

    # --- 叠加标注“配对的 high->low 段”（受 lookback、min_drop、min_bars_per_leg 等影响）
    # 先筛出两端点都在绘图窗口内的配对段
    in_tail = [
        (hi_t, hi_v, lo_t, lo_v, drop)
        for (hi_t, hi_v, lo_t, lo_v, drop) in pairs
        if (hi_t in df_tail.index) and (lo_t in df_tail.index)
    ]

    # 只画窗口内最近10段；如需全部可用 `for ... in in_tail:`
    for (hi_t, hi_v, lo_t, lo_v, drop) in in_tail[-10:]:
        # 1) 连线 + 端点
        plt.plot([hi_t, lo_t], [hi_v, lo_v], linewidth=1.0, alpha=0.9, color="C1")
        plt.scatter([hi_t, lo_t], [hi_v, lo_v], s=45, color="C1")

        # 2) 在线段中点标注回撤百分比
        mid_x = hi_t + (lo_t - hi_t) / 2
        mid_y = (hi_v + lo_v) / 2
        txt = f"-{drop * 100:.1f}%"

        # 根据线段方向，微调文字上下偏移，避免压线
        offset_y = -6 if hi_v >= lo_v else 6
        plt.annotate(
            txt,
            xy=(mid_x, mid_y),
            xytext=(0, offset_y), textcoords="offset points",
            ha="center", va="center", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
        )

    # --- 标 pivot（若有效）
    if chk["valid"]:
        pv_t, pv_p = chk["pivot_time"], chk["pivot_price"]
        if pv_t in df_tail.index:
            plt.axhline(pv_p, linestyle="--", linewidth=1, color="C3", label=f"Pivot ~ {pv_p:.2f}")

    plt.title(f"{symbol} VCP legs (recent)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("/home/ubuntu/plot/swing_high_low_pivot.png")
    plt.close()

except Exception as e:
    print("Plot skipped:", e)
