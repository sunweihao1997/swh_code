"""
2025-09-07
Test2: Continue from Line120 in Test1 since no pairs found

This script is used to test the VCP strategy implementation in pybroker.

Three sub-scripts:
`cal_stock_ATR_shrinking_250901.py`
`cal_stock_VCP_Algorithm_v2_250904.py`
`cal_stock_volume_energy_break_v2_250904.py`
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cal_stock_ATR_shrinking_250901 import (
    atr_ta, bb_width_ta, ensure_datetime_index, link_week_daily,
    to_weekly, percentile_rank_last, low_volatility_flags
)
from cal_stock_VCP_Algorithm_v2_250904 import (
    find_swings_zigzag, extract_contractions_v2, is_valid_vcp_sequence_v2
)

# =============== 工具：强制把 df 变成 DatetimeIndex（并删除遗留的 'date' 列） ===============
def force_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    # 如果已有 DatetimeIndex，仅确保升序 & 去重
    if isinstance(_df.index, pd.DatetimeIndex):
        _df = _df[~_df.index.duplicated(keep="last")].sort_index()
    else:
        if "date" not in _df.columns:
            raise ValueError("需要 'date' 列来设置 DatetimeIndex。")
        _df["date"] = pd.to_datetime(_df["date"], errors="coerce")
        _df = (
            _df.dropna(subset=["date"])
               .drop_duplicates(subset=["date"], keep="last")
               .set_index("date")
               .sort_index()
        )
    # 有些流程（例如 ensure_datetime_index 的旧实现）会保留 'date' 列，这里一律删除
    if "date" in _df.columns:
        _df = _df.drop(columns=["date"])
    return _df

# ================= The test stock ===================
stock_code = "002245"
stock_path = f"/home/sun/data_n100/stocks_price/{stock_code}.xlsx"

# -------- 0) 读取并“强制”设为 DatetimeIndex（关键修复） --------
raw = pd.read_excel(stock_path, parse_dates=True)
stock_data = force_datetime_index(raw)

# 列名统一为小写（防止大小写不一致）
stock_data.columns = [c.lower() for c in stock_data.columns]

# ================= 1. ATR 收缩（周-日联动） ===================
flags = low_volatility_flags(df_daily=stock_data)
daily_flags = flags["daily"]
weekly_flags = flags["weekly"]

candidates = link_week_daily(
    daily_df=stock_data,
    flags_d=daily_flags,
    flags_w=weekly_flags,
    exec_window_days=30,
    start_next_day=True
)
# 保险：候选序列索引也保证是 DatetimeIndex（一般已经是）
candidates.index = pd.to_datetime(candidates.index)

# ================= 1.2 Plot ===================
def plot_candidates(df, candidates, title="Stock Price with Candidate Days"):
    d = df.copy()
    close = d["close"]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(close.index, close.values, label="Close Price", linewidth=1.5)

    cand_dates = candidates[candidates].index.intersection(close.index)
    cand_prices = close.loc[cand_dates]
    ax.scatter(cand_prices.index, cand_prices.values, s=50, label="Candidate Days", zorder=5, marker="o")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    os.makedirs("/home/sun/paint/stock", exist_ok=True)
    plt.savefig(f"/home/sun/paint/stock/{stock_code}_candidates.png")

# ================= 2. VCP ===================

# 2.1 直接按索引 join 候选列（不会破坏 DatetimeIndex）
stock_data = stock_data.join(candidates.rename("candidate"))

# 2.2 （再次兜底）确保此刻仍是 DatetimeIndex
stock_data = force_datetime_index(stock_data)

# 2.3 计算 ZigZag（**和价格索引天然对齐**）
zigzag_result = find_swings_zigzag(
    df=stock_data,
    threshold=0.03,        # 先取 3%，保证能打到点；后续可回调 0.04/0.05
    mode="percent",
    use_high_low=True,
    include_last=True
)
# 双保险对齐
zigzag_result = zigzag_result.reindex(stock_data.index)

# 2.4 提取回撤段（先放宽参数，确保能产出）
res_pairs = extract_contractions_v2(
    df=stock_data,
    swings=zigzag_result,
    lookback=None,
    min_drop=0.05,         # 先 2%
    max_drop=0.45,         # 放宽上限
    min_bars_per_leg=5,    # 先 3 根
    max_last_drop=None
)
print(f"[VCP] pairs={len(res_pairs['pairs'])}, last_drop={res_pairs['last_drop']}")

# 2.5 VCP 校验（ATR 自适应低点抬高）
atr20 = atr_ta(stock_data, n=20)
chk = is_valid_vcp_sequence_v2(
    res_pairs,
    need_n=2,
    tol_drop=0.0,
    enforce_higher_lows=True,
    low_mode="atr",
    low_abs=0.0,
    low_pct=0.0,
    low_atr_mult=0.5,
    atr_series=atr20,
    require_last_tight=False
)
print(f"[VCP] valid={chk['valid']} reason={chk['reason']}")
if chk["valid"]:
    print("drops:", chk["drops"])
    print("lows:", chk["lows"])
    print("pivot:", chk["pivot_time"], chk["pivot_price"])

# 2.6 叠图
def plot_vcp_signals(df, zigzag_swings, chk, title="VCP Signals (ZigZag)"):
    d = df.copy()
    close = d["close"]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(close.index, close.values, label="Close", linewidth=1.2)

    zz = zigzag_swings.dropna(how="all")
    if not zz.empty:
        highs = zz["swing_high"].dropna()
        lows  = zz["swing_low"].dropna()
        ax.scatter(highs.index, close.loc[highs.index], s=20, marker="^", label="ZZ High", alpha=0.6)
        ax.scatter(lows.index,  close.loc[lows.index],  s=20, marker="v", label="ZZ Low",  alpha=0.6)

    if chk.get("valid", False) and chk.get("pivot_price") is not None:
        t = chk["pivot_time"]
        p = chk["pivot_price"]
        ax.hlines(p, xmin=t, xmax=t + pd.Timedelta(days=1), linestyles="dotted", linewidth=1.2, label=f"Pivot {p:.2f}")

    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    os.makedirs("/home/sun/paint/stock", exist_ok=True)
    plt.savefig(f"/home/sun/paint/stock/{stock_code}_vcp_signals.png")

# =============== 诊断打印（确认索引已修复） ===============
def print_basic_diagnostics():
    print("INDEX:", stock_data.index.dtype, stock_data.index.is_monotonic_increasing, stock_data.index.has_duplicates)
    print("columns:", list(stock_data.columns))
    print("swings aligned?", zigzag_result.index.equals(stock_data.index))
    print("swing_high count:", zigzag_result["swing_high"].notna().sum())
    print("swing_low  count:", zigzag_result["swing_low"].notna().sum())
    sh = zigzag_result["swing_high"].dropna()
    sl = zigzag_result["swing_low"].dropna()
    if not sh.empty:
        print("first non-NaN high at:", sh.index.min())
        print("last  non-NaN high at:", sh.index.max())
    if not sl.empty:
        print("first non-NaN  low at:", sl.index.min())
        print("last  non-NaN  low at:", sl.index.max())

def plot_vcp_signals_with_pairs(
    df: pd.DataFrame,
    zigzag_swings: pd.DataFrame,
    res_pairs: dict,
    chk: dict | None = None,
    *,
    max_pairs_to_draw: int = 60,   # 最多画多少段，防止太拥挤
    highlight_last_n: int = 3,     # 高亮最近 N 段
    annotate_drop: bool = True,    # 在线段上标注回撤比例
    title: str = "VCP Signals + High→Low Pairs",
    outpath: str | None = None
):
    """
    在收盘价曲线上叠加：
      - ZigZag 拐点（^ / v）
      - 配对的 high→low 段（带回撤标注）
      - （可选）VCP 枢轴水平线
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import numpy as np

    d = df.copy()
    if "date" in d.columns:
        d["date"] = pd.to_datetime(d["date"])
        d = d.set_index("date")
    close = d["close"]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(close.index, close.values, label="Close", linewidth=1.2)

    # 1) 画 ZigZag 拐点
    zz = zigzag_swings.dropna(how="all")
    if not zz.empty:
        highs = zz["swing_high"].dropna()
        lows  = zz["swing_low"].dropna()
        if not highs.empty:
            ax.scatter(highs.index, close.loc[highs.index], s=24, marker="^", label="ZZ High", alpha=0.65)
        if not lows.empty:
            ax.scatter(lows.index,  close.loc[lows.index],  s=24, marker="v", label="ZZ Low",  alpha=0.65)

    # 2) 画 high→low 段
    pairs = res_pairs.get("pairs", []) if isinstance(res_pairs, dict) else (res_pairs or [])
    n_pairs = len(pairs)

    if n_pairs > 0:
        # 只画最近 max_pairs_to_draw 段，避免太密
        start_idx = max(0, n_pairs - max_pairs_to_draw)
        view_pairs = pairs[start_idx:]

        for j, (hi_t, hi_v, lo_t, lo_v, drop) in enumerate(view_pairs):
            # 基础线
            ax.plot([hi_t, lo_t], [hi_v, lo_v],
                    linewidth=1.2, alpha=0.9 if j >= len(view_pairs)-highlight_last_n else 0.5,
                    linestyle='-', label=None)

            # 标注回撤比例
            if annotate_drop:
                mid_t = hi_t + (lo_t - hi_t) / 2
                mid_p = (hi_v + lo_v) / 2
                try:
                    ax.text(mid_t, mid_p, f"{drop*100:.1f}%", fontsize=9, ha='center', va='bottom', alpha=0.8)
                except Exception:
                    # 某些 matplotlib 版本对时间轴 text 可能挑剔，回退到略微偏右
                    ax.text(lo_t, mid_p, f"{drop*100:.1f}%", fontsize=9, ha='left', va='bottom', alpha=0.8)

        # 高亮最近 N 段的端点，便于视觉定位
        if highlight_last_n > 0:
            for (hi_t, hi_v, lo_t, lo_v, drop) in view_pairs[-highlight_last_n:]:
                ax.scatter([hi_t, lo_t], [hi_v, lo_v], s=36, edgecolor='k', linewidths=0.6, zorder=5)

    # 3) VCP 枢轴水平线
    if chk and chk.get("valid") and chk.get("pivot_price") is not None:
        p = float(chk["pivot_price"])
        t = chk.get("pivot_time")
        # 画一天长度的虚线，避免整条水平线遮挡图面
        if isinstance(t, pd.Timestamp):
            ax.hlines(p, xmin=t, xmax=t + pd.Timedelta(days=1), linestyles="dotted", linewidth=1.2, label=f"Pivot {p:.2f}")
        else:
            ax.axhline(p, linestyle="dotted", linewidth=1.2, label=f"Pivot {p:.2f}")

    # 4) 轴样式
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    if outpath is not None:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath)
    else:
        plt.show()


# ================= Main =================
if __name__ == "__main__":
    print_basic_diagnostics()
    plot_candidates(stock_data, candidates, title=f"Stock {stock_code} Price with Candidate Days")
    plot_vcp_signals(stock_data, zigzag_result, chk, title=f"Stock {stock_code} VCP Signals (ZigZag)")
    # …前面已算好 zigzag_result, res_pairs, chk …

    plot_vcp_signals_with_pairs(
        df=stock_data,
        zigzag_swings=zigzag_result,
        res_pairs=res_pairs,
        chk=chk,
        max_pairs_to_draw=80,                # 想多看几段可调大
        highlight_last_n=3,                  # 高亮最近3段
        annotate_drop=True,
        title=f"Stock {stock_code} — VCP + High→Low Pairs",
        outpath=f"/home/sun/paint/stock/{stock_code}_vcp_pairs.png"
    )

