# -*- coding: utf-8 -*-
"""
20250903
This script is to test cal_stock_VCP_Algorithm_250902 with breakout trigger
"""

# ================== 基本依赖 ==================
import pandas as pd
import numpy as np
import akshare as ak
import matplotlib.pyplot as plt

from typing import List, Tuple, Optional, Union, Dict

# ===== 你的算法函数：若在同目录另一文件中，请保持模块名一致 =====
#   包含：find_swings, extract_contractions_v2, is_valid_vcp_sequence_v2
from cal_stock_VCP_Algorithm_250902 import (
    find_swings, extract_contractions_v2, is_valid_vcp_sequence_v2
)

# ================== 参数区域（按需修改） ==================
symbol      = "603156"      # 测试标的（A股6位代码）
start_date  = "20200101"
end_date    = "20250902"
adjust      = "qfq"         # "qfq"前复权 / ""不复权 / "hfq"后复权

# VCP识别参数
SWING_WINDOW        = 5
LOOKBACK            = 2000
MIN_DROP            = 0.01
MAX_DROP            = 0.35
MIN_BARS_PER_LEG    = 3
MAX_LAST_DROP       = 0.15

# VCP序列校验参数
NEED_N              = 3
TOL_DROP            = 0.01
ENFORCE_HIGHER_LOWS = True
LOW_TOL             = 0.0
REQUIRE_LAST_TIGHT  = True

# 突破触发（价量）
BRK_PRICE_BUF   = 0.01   # 价差缓冲：≥ Pivot*(1+buf)
BRK_VOL_MULT    = 1.8    # 放量门槛：≥ 1.8×MA50(Volume)
BRK_LOOKAHEAD   = 20     # Pivot 后观察天数

# 画图
TAIL_N      = 500
SAVE_PATH   = "/home/ubuntu/plot/swing_high_low_pivot.png"

# ================== 函数：突破触发 ==================
def find_breakout_signals(df: pd.DataFrame,
                          pivot_time: pd.Timestamp,
                          pivot_price: float,
                          buf: float = 0.01,
                          vol_mult: float = 1.8,
                          look_ahead_days: int = 20) -> pd.DataFrame:
    """
    在 pivot_time 之后 look_ahead_days 内，寻找突破枢轴的买点：
      条件1：Close >= pivot_price * (1+buf)
      条件2：Volume >= vol_mult * MA50(Volume)
    返回满足条件的日期和关键字段（索引为突破日期）。
    """
    if pivot_time in df.index:
        start_idx = df.index.get_loc(pivot_time)
    else:
        start_idx = df.index.searchsorted(pivot_time)

    start_idx = min(start_idx + 1, len(df) - 1)
    end_idx = min(start_idx + look_ahead_days, len(df) - 1)
    if start_idx >= end_idx:
        return pd.DataFrame()

    seg = df.iloc[start_idx:end_idx + 1].copy()
    seg["VolMA50"] = df["Volume"].rolling(50, min_periods=10).mean().reindex(seg.index)

    seg["price_ok"] = seg["Close"] >= pivot_price * (1.0 + buf)
    seg["vol_ok"]   = seg["Volume"] >= vol_mult * seg["VolMA50"]
    seg["breakout"] = seg["price_ok"] & seg["vol_ok"]

    return seg.loc[seg["breakout"], ["Close", "Volume", "VolMA50", "price_ok", "vol_ok"]]


# ================== 主流程 ==================
if __name__ == "__main__":
    # 1) 拉取数据
    df_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start_date, end_date=end_date, adjust=adjust)

    # 2) 清洗为标准OHLCV
    df = df_raw.rename(columns={
        "日期": "Date", "开盘": "Open", "最高": "High",
        "最低": "Low", "收盘": "Close", "成交量": "Volume"
    }).copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 若需把“手”换成“股”，可启用：
    # df["Volume"] = df["Volume"] * 100

    # 3) VCP收缩检测
    sw = find_swings(df, window=SWING_WINDOW)

    res = extract_contractions_v2(
        df, sw,
        lookback=LOOKBACK,
        min_drop=MIN_DROP,
        max_drop=MAX_DROP,
        min_bars_per_leg=MIN_BARS_PER_LEG,
        max_last_drop=MAX_LAST_DROP
    )

    pairs = res["pairs"]
    print(f"\n=== {symbol} 回撤段（近{len(pairs)}段，显示最近5段）===")
    for tup in pairs[-5:]:
        hi_t, hi_v, lo_t, lo_v, drop = tup
        print(f"hi@{hi_t.date()}={hi_v:.2f}  →  low@{lo_t.date()}={lo_v:.2f}  drop={drop*100:.2f}%")

    # 3.3 校验 VCP 收缩序列
    chk = is_valid_vcp_sequence_v2(
        res,
        need_n=NEED_N,
        tol_drop=TOL_DROP,
        enforce_higher_lows=ENFORCE_HIGHER_LOWS,
        low_tol=LOW_TOL,
        require_last_tight=REQUIRE_LAST_TIGHT
    )

    print("\n=== 校验结果 ===")
    print("valid:", chk["valid"])
    print("reason:", chk["reason"])
    print("drops (最近N段):", chk["drops"])
    print("lows  (最近N段):", chk["lows"])
    print("last_drop:", chk["last_drop"])
    print("passed_last_tight:", chk["passed_last_tight"])
    print("pivot_time:", chk["pivot_time"], "pivot_price:", chk["pivot_price"])

    # 3.4 突破触发（价量）
    buy_signals = pd.DataFrame()
    if chk["valid"] and (chk["pivot_time"] is not None):
        buy_signals = find_breakout_signals(
            df,
            pivot_time=chk["pivot_time"],
            pivot_price=chk["pivot_price"],
            buf=BRK_PRICE_BUF,
            vol_mult=BRK_VOL_MULT,
            look_ahead_days=BRK_LOOKAHEAD
        )

    print("\n=== 突破买点（若有） ===")
    if buy_signals.empty:
        print("未触发价量突破（或 VCP 不成立）。")
    else:
        print(buy_signals)

    # 4) 可视化：收盘、Swing、配对段+回撤%、Pivot、突破点
    try:
        tail_n  = TAIL_N
        df_tail = df.tail(tail_n).copy()
        sw_tail = sw.loc[df_tail.index]

        plt.figure(figsize=(11, 5))
        plt.plot(df_tail.index, df_tail["Close"], label="Close", linewidth=1.2)

        # Swing ▲/▼
        hi_mask = ~pd.isna(sw_tail["swing_high"])
        lo_mask = ~pd.isna(sw_tail["swing_low"])
        plt.scatter(df_tail.index[hi_mask], sw_tail.loc[hi_mask, "swing_high"],
                    marker="^", s=60, label="Swing High")
        plt.scatter(df_tail.index[lo_mask], sw_tail.loc[lo_mask, "swing_low"],
                    marker="v", s=60, label="Swing Low")

        # 配对段（筛在窗口内），并标注回撤百分比
        in_tail = [
            (hi_t, hi_v, lo_t, lo_v, drop)
            for (hi_t, hi_v, lo_t, lo_v, drop) in pairs
            if (hi_t in df_tail.index) and (lo_t in df_tail.index)
        ]
        for (hi_t, hi_v, lo_t, lo_v, drop) in in_tail[-10:]:  # 只画最近10段，改为 in_tail 画全量
            plt.plot([hi_t, lo_t], [hi_v, lo_v], linewidth=1.0, alpha=0.9, color="C1")
            plt.scatter([hi_t, lo_t], [hi_v, lo_v], s=45, color="C1")

            # 中点标注 drop%
            mid_x = hi_t + (lo_t - hi_t) / 2
            mid_y = (hi_v + lo_v) / 2
            txt   = f"-{drop * 100:.1f}%"
            offset_y = -6 if hi_v >= lo_v else 6
            plt.annotate(
                txt, xy=(mid_x, mid_y), xytext=(0, offset_y), textcoords="offset points",
                ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.6)
            )

        # Pivot 水平线
        if chk["valid"]:
            pv_t, pv_p = chk["pivot_time"], chk["pivot_price"]
            if pv_t in df_tail.index:
                plt.axhline(pv_p, linestyle="--", linewidth=1, color="C3", label=f"Pivot ~ {pv_p:.2f}")

        # 突破点（绿点 + 竖虚线，仅标在 tail 窗口内的）
        if not buy_signals.empty:
            bs_tail = buy_signals.loc[buy_signals.index.isin(df_tail.index)]
            if not bs_tail.empty:
                plt.scatter(bs_tail.index, bs_tail["Close"], s=55, marker="o",
                            color="tab:green", label="Breakout")
                for dt, _ in bs_tail.iterrows():
                    plt.axvline(dt, linestyle="--", linewidth=0.8, color="tab:green", alpha=0.5)

        plt.title(f"{symbol} VCP legs (recent)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(SAVE_PATH)
        plt.close()
        print(f"\n图已保存：{SAVE_PATH}")
    except Exception as e:
        print("Plot skipped:", e)
