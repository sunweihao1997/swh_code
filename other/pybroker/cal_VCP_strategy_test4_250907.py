"""
2025-09-07 最终版
- 修复 DatetimeIndex & 列名统一
- ZigZag 拐点 → 回撤段 → VCP 校验
- 扫描“过去5年”所有 VCP 窗口并可视化（高亮区间 + 枢轴线）
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cal_stock_ATR_shrinking_250901 import (
    atr_ta, low_volatility_flags, link_week_daily
)
from cal_stock_VCP_Algorithm_v2_250904 import (
    find_swings_zigzag, extract_contractions_v2, is_valid_vcp_sequence_v2
)

# =============== 基础工具：强制 DatetimeIndex ===============
def force_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
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
    if "date" in _df.columns:  # 删除遗留列避免后续误用
        _df = _df.drop(columns=["date"])
    return _df

# =============== 画候选日 ===============
def plot_candidates(df, candidates, stock_code, title="Stock Price with Candidate Days"):
    d = df.copy()
    close = d["close"]

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(close.index, close.values, label="Close", linewidth=1.3)

    cand_dates = candidates[candidates].index.intersection(close.index)
    if len(cand_dates) > 0:
        ax.scatter(cand_dates, close.loc[cand_dates], s=46, label="Candidate", zorder=5, marker="o")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend()
    plt.tight_layout()
    os.makedirs("/home/sun/paint/stock", exist_ok=True)
    plt.savefig(f"/home/sun/paint/stock/{stock_code}_candidates.png")

# =============== 扫描历史窗口（过去5年） ===============
def find_all_vcp_windows(
    pairs: list,
    atr_series: pd.Series,
    *,
    need_n: int = 3,
    tol_drop: float = 0.0,
    enforce_higher_lows: bool = True,
    low_mode: str = "atr",
    low_abs: float = 0.0,
    low_pct: float = 0.0,
    low_atr_mult: float = 0.5,
    require_last_tight: bool = False,
) -> list[dict]:
    """在全部 pairs 上滑窗检查，返回所有 valid 的窗口信息。"""
    out = []
    if len(pairs) < need_n:
        return out
    for i in range(len(pairs) - need_n + 1):
        sub = pairs[i:i+need_n]
        chk = is_valid_vcp_sequence_v2(
            sub,
            need_n=need_n,
            tol_drop=tol_drop,
            enforce_higher_lows=enforce_higher_lows,
            low_mode=low_mode, low_abs=low_abs, low_pct=low_pct,
            low_atr_mult=low_atr_mult, atr_series=atr_series,
            require_last_tight=require_last_tight
        )
        if chk["valid"]:
            out.append({
                "start": sub[0][0],          # 第一个 swing high 的时间
                "end":   sub[-1][2],         # 最后一个 swing low 的时间
                "pivot_time": chk["pivot_time"],
                "pivot_price": chk["pivot_price"],
                "drops": chk["drops"],
                "lows":  chk["lows"],
                "sub_pairs": sub,
                "win_index": i
            })
    return out

# =============== 可视化：ZigZag + pairs + VCP 窗口高亮 ===============
def plot_vcp_signals_with_pairs_and_windows(
    df: pd.DataFrame,
    zigzag_swings: pd.DataFrame,
    res_pairs: dict,
    windows: list[dict],
    stock_code: str,
    title: str = "VCP — Pairs & Valid Windows (Last 5y)",
    outpath: str | None = None,
    max_pairs_to_draw: int = 120,
):
    d = df.copy()
    close = d["close"]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(close.index, close.values, label="Close", linewidth=1.2)

    # ZigZag 拐点
    zz = zigzag_swings.dropna(how="all")
    highs = zz["swing_high"].dropna()
    lows  = zz["swing_low"].dropna()
    if not highs.empty:
        ax.scatter(highs.index, close.loc[highs.index], s=22, marker="^", label="ZZ High", alpha=0.6)
    if not lows.empty:
        ax.scatter(lows.index,  close.loc[lows.index],  s=22, marker="v", label="ZZ Low",  alpha=0.6)

    # high→low 段
    pairs = res_pairs.get("pairs", [])
    if pairs:
        draw_pairs = pairs[-max_pairs_to_draw:]
        for (hi_t, hi_v, lo_t, lo_v, drop) in draw_pairs:
            ax.plot([hi_t, lo_t], [hi_v, lo_v], linewidth=1.0, alpha=0.65)

    # 高亮所有 valid 窗口（半透明）
    for w in windows:
        ax.axvspan(w["start"], w["end"], color="tab:green", alpha=0.12)
        # 在 pivot_time 画短虚线
        pt, pp = w["pivot_time"], w["pivot_price"]
        if pd.notna(pp):
            if isinstance(pt, pd.Timestamp):
                ax.hlines(pp, xmin=pt, xmax=pt + pd.Timedelta(days=1),
                          linestyles="dotted", linewidth=1.1, label=None)

    # 图例与样式
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        plt.savefig(outpath)
    else:
        plt.show()

# ============================== 主流程 ==============================
if __name__ == "__main__":
    # ---------- 基础参数 ----------
    stock_code = "002245"
    stock_path = f"/home/sun/data_n100/stocks_price/{stock_code}.xlsx"

    # ---------- 读取 & 统一索引 ----------
    raw = pd.read_excel(stock_path, parse_dates=True)
    stock_data = force_datetime_index(raw)
    stock_data.columns = [c.lower() for c in stock_data.columns]  # 统一小写
    # 必要列检查
    for col in ["high", "low", "close"]:
        if col not in stock_data.columns:
            raise ValueError(f"缺少必需列: {col}")

    # ---------- （可选）ATR收缩周-日联动候选 ----------
    flags = low_volatility_flags(df_daily=stock_data)
    daily_flags = flags["daily"]
    weekly_flags = flags["weekly"]
    candidates = link_week_daily(
        daily_df=stock_data,
        flags_d=daily_flags,
        flags_w=weekly_flags,
        exec_window_days=20,
        start_next_day=True
    )
    # 只用于参考绘图，不参与 VCP 判定
    stock_data = stock_data.join(candidates.rename("candidate"))

    # ---------- ZigZag 拐点 ----------
    zigzag_result = find_swings_zigzag(
        df=stock_data,
        threshold=0.05,          # 可调：0.03~0.05
        mode="percent",
        use_high_low=True,
        include_last=True
    ).reindex(stock_data.index)

    # ---------- 提取回撤段 ----------
    res_pairs = extract_contractions_v2(
        df=stock_data,
        swings=zigzag_result,
        lookback=None,
        min_drop=0.03,           # 先稍宽，验证产出；可回调到 0.03
        max_drop=0.45,
        min_bars_per_leg=5,      # 先 3 根；可回调到 5~7
        max_last_drop=None
    )
    print(f"[VCP] pairs={len(res_pairs['pairs'])}, last_drop={res_pairs['last_drop']}")

    # ---------- 先看“最近窗口”是否满足（诊断用） ----------
    atr20 = atr_ta(stock_data, n=20)
    recent_chk = is_valid_vcp_sequence_v2(
        res_pairs, need_n=2, tol_drop=0.0,
        enforce_higher_lows=True,
        low_mode="atr", low_atr_mult=0.5, atr_series=atr20,
        require_last_tight=False
    )
    print(f"[VCP] recent valid={recent_chk['valid']} reason={recent_chk['reason']}")

    # ---------- 扫描过去 5 年的所有窗口 ----------
    if len(stock_data.index) == 0:
        raise RuntimeError("stock_data 为空。")
    end_date = stock_data.index.max()
    start_5y = end_date - pd.DateOffset(years=5)

    all_windows = find_all_vcp_windows(
        res_pairs["pairs"],
        atr_series=atr20,
        need_n=2,
        tol_drop=0.0,
        enforce_higher_lows=True,
        low_mode="atr",
        low_abs=0.0,
        low_pct=0.0,
        low_atr_mult=0.5,
        require_last_tight=False,
    )

    # 仅保留过去5年落入区间的窗口（窗口结束时间在 5 年内）
    windows_5y = [w for w in all_windows if (w["end"] >= start_5y)]

    print(f"[VCP] valid windows in last 5 years: {len(windows_5y)}")
    for w in windows_5y:
        print(f"  - {w['start'].date()} → {w['end'].date()}  pivot={w['pivot_price']:.2f}")

    # ---------- 可视化 ----------
    # 价格 + ZigZag 拐点 + 所有 pairs + 过去5年满足的窗口（绿色半透明高亮）+ pivot 短虚线
    out_img = f"/home/sun/paint/stock/{stock_code}_vcp_pairs_windows_5y.png"
    plot_vcp_signals_with_pairs_and_windows(
        df=stock_data,
        zigzag_swings=zigzag_result,
        res_pairs=res_pairs,
        windows=windows_5y,
        stock_code=stock_code,
        title=f"{stock_code} — VCP Pairs & Valid Windows (Last 5y)",
        outpath=out_img,
        max_pairs_to_draw=120
    )

    # ---------- 终端摘要 ----------
    print("\n===== SUMMARY =====")
    print("INDEX:", stock_data.index.dtype, stock_data.index.is_monotonic_increasing, stock_data.index.has_duplicates)
    print("swings aligned?", zigzag_result.index.equals(stock_data.index))
    print("swing_high count:", zigzag_result["swing_high"].notna().sum(),
          "swing_low count:", zigzag_result["swing_low"].notna().sum())
    sh = zigzag_result["swing_high"].dropna()
    sl = zigzag_result["swing_low"].dropna()
    if not sh.empty and not sl.empty:
        print("range:", sh.index.min().date(), "→", sh.index.max().date())
    print(f"Saved figure: {out_img}")
