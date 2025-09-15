# -*- coding: utf-8 -*-
"""
Scan A-shares for VCP + breakout in the last 5 years.
Saves: scan_summary.csv, scan_events.csv
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import akshare as ak

from typing import List, Tuple, Optional, Union, Dict

# === 你已有的函数（确保模块名正确） ===
from cal_stock_VCP_Algorithm_250902 import (
    find_swings, extract_contractions_v2, is_valid_vcp_sequence_v2
)

# ================== 参数区（按需调整） ==================
YEARS_BACK          = 5
ADJUST              = "qfq"      # "qfq"前复权 / ""不复权 / "hfq"后复权
MIN_BARS_PER_SYMBOL = 252        # 最少交易日数据，太少直接跳

# VCP识别参数（与你平时测试保持一致）
SWING_WINDOW        = 5          # swing 抗噪：5~11 合理
LOOKBACK_FOR_PAIRS  = None       # None = 全量；否则限定近 N 根内配对
MIN_DROP            = 0.04
MAX_DROP            = 0.40
MIN_BARS_PER_LEG    = 5
MAX_LAST_DROP       = 0.15

# 序列校验（最近 need_n 段）
NEED_N              = 3
TOL_DROP            = 0.01
ENFORCE_HIGHER_LOWS = True
LOW_TOL             = 0.0
REQUIRE_LAST_TIGHT  = True

# 突破触发（价量）
BRK_PRICE_BUF       = 0.01       # 价 ≥ Pivot*(1+buf)
BRK_VOL_MULT        = 1.8        # 量 ≥ 1.8×MA50(Vol)
BRK_LOOKAHEAD       = 20         # Pivot 之后观察天数

# 扫描控制
MAX_SYMBOLS         = None       # None=全量；否则限制前 N 只用于试跑
SLEEP_SEC_EACH      = 0.2        # 每票间隔（akshare 友好）

# 输出
OUT_SUMMARY_CSV     = "scan_summary.csv"
OUT_EVENTS_CSV      = "scan_events.csv"


# ================== 工具函数 ==================
def yyyymmdd(dt: pd.Timestamp) -> str:
    return dt.strftime("%Y-%m-%d")


def load_a_share_symbols(max_symbols: Optional[int] = None) -> pd.DataFrame:
    """
    返回含代码/名称的数据框：列 ['代码','名称']。
    """
    spot = ak.stock_zh_a_spot_em()
    spot = spot[['代码', '名称']].copy()
    spot = spot.dropna()
    if max_symbols is not None:
        spot = spot.head(max_symbols)
    return spot


def fetch_history(symbol: str, start_date: str, end_date: str, adjust: str = ADJUST) -> pd.DataFrame:
    """
    拉取单票日线并清洗为标准 OHLCV（Date 索引）
    """
    df_raw = ak.stock_zh_a_hist(symbol=symbol, period="daily",
                                start_date=start_date, end_date=end_date, adjust=adjust)
    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = df_raw.rename(columns={
        "日期": "Date", "开盘": "Open", "最高": "High",
        "最低": "Low", "收盘": "Close", "成交量": "Volume"
    }).copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 你若想把“手”换“股”，可启用：
    # df["Volume"] = df["Volume"] * 100
    return df.dropna(subset=["Open","High","Low","Close","Volume"])


def find_breakout_signals(df: pd.DataFrame,
                          pivot_time: pd.Timestamp,
                          pivot_price: float,
                          buf: float = BRK_PRICE_BUF,
                          vol_mult: float = BRK_VOL_MULT,
                          look_ahead_days: int = BRK_LOOKAHEAD) -> pd.DataFrame:
    """
    在 pivot_time 之后寻找突破买点（价量双确认）：
      价：Close >= pivot*(1+buf)
      量：Volume >= vol_mult * MA50(Volume)
    返回满足条件的行（索引=突破日）
    """
    if pivot_time in df.index:
        start_idx = df.index.get_loc(pivot_time)
    else:
        start_idx = df.index.searchsorted(pivot_time)

    start_idx = min(start_idx + 1, len(df) - 1)
    end_idx   = min(start_idx + look_ahead_days, len(df) - 1)
    if start_idx >= end_idx:
        return pd.DataFrame()

    seg = df.iloc[start_idx:end_idx + 1].copy()
    seg["VolMA50"] = df["Volume"].rolling(50, min_periods=10).mean().reindex(seg.index)

    seg["price_ok"] = seg["Close"] >= pivot_price * (1.0 + buf)
    seg["vol_ok"]   = seg["Volume"] >= vol_mult * seg["VolMA50"]
    seg["breakout"] = seg["price_ok"] & seg["vol_ok"]

    return seg.loc[seg["breakout"], ["Close", "Volume", "VolMA50", "price_ok", "vol_ok"]]


def scan_symbol_for_vcp_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    """
    在整个历史（近 N 年）上扫描“所有可能的 VCP 形态 + 突破”。
    做法：一次性配出全量 high->low pairs，然后沿 pairs 序列逐步验证最近 need_n 段是否满足 VCP；
         每次满足即在 pivot 后窗口内找突破事件。返回该票的所有事件（逐行）。
    """
    if df.shape[0] < MIN_BARS_PER_SYMBOL:
        return pd.DataFrame()

    # 1) 摆动点
    swings = find_swings(df, window=SWING_WINDOW)

    # 2) 一次性配对 high->low（尽量覆盖全历史）
    lookback = LOOKBACK_FOR_PAIRS or len(df)
    res_full = extract_contractions_v2(
        df, swings,
        lookback=lookback,
        min_drop=MIN_DROP,
        max_drop=MAX_DROP,
        min_bars_per_leg=MIN_BARS_PER_LEG,
        max_last_drop=MAX_LAST_DROP
    )
    pairs: List[Tuple[pd.Timestamp, float, pd.Timestamp, float, float]] = res_full["pairs"]
    if len(pairs) < NEED_N:
        return pd.DataFrame()

    events = []
    # 遍历到每个位置 k，检查“最近 need_n 段”是否构成 VCP
    for k in range(NEED_N - 1, len(pairs)):
        sub = pairs[:k + 1]  # 到第 k 段为止的历史
        chk = is_valid_vcp_sequence_v2(
            sub,
            need_n=NEED_N,
            tol_drop=TOL_DROP,
            enforce_higher_lows=ENFORCE_HIGHER_LOWS,
            low_tol=LOW_TOL,
            require_last_tight=REQUIRE_LAST_TIGHT
        )
        if not chk["valid"] or chk["pivot_time"] is None:
            continue

        # 3) 在 pivot 之后寻找突破事件
        brks = find_breakout_signals(
            df,
            pivot_time=chk["pivot_time"],
            pivot_price=chk["pivot_price"],
            buf=BRK_PRICE_BUF,
            vol_mult=BRK_VOL_MULT,
            look_ahead_days=BRK_LOOKAHEAD
        )
        if brks.empty:
            continue

        # 记录所有突破日（如只想记录第一个，可用 brks.head(1)）
        for dt, row in brks.iterrows():
            events.append({
                "break_date": dt,
                "pivot_time": chk["pivot_time"],
                "pivot_price": float(chk["pivot_price"]),
                "close": float(row["Close"]),
                "vol": float(row["Volume"]),
                "vol_ma50": float(row["VolMA50"]),
                "last_drop": float(chk["last_drop"]) if chk["last_drop"] is not None else np.nan,
                "drops_seq": chk["drops"],    # 最近 need_n 段的回撤列表
                "lows_seq": chk["lows"],      # 最近 need_n 段的低点列表
            })

    if not events:
        return pd.DataFrame()
    return pd.DataFrame(events).sort_values("break_date")


def scan_a_share_vcp_breakouts() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    扫描全部A股；返回 (summary_df, events_df)。
    summary_df: 每票聚合（触发次数、首/末次触发日期）
    events_df : 逐事件明细
    """
    # 计算时间范围
    end_dt = datetime.today()
    start_dt = end_dt - timedelta(days=365 * YEARS_BACK)
    start_str = start_dt.strftime("%Y%m%d")
    end_str   = end_dt.strftime("%Y%m%d")

    # 取代码表
    codes_df = load_a_share_symbols(MAX_SYMBOLS)
    print(f"将扫描 {len(codes_df)} 只股票（过去 {YEARS_BACK} 年）...")

    all_events = []
    t0 = time.time()

    for i, row in codes_df.iterrows():
        code = str(row["代码"])
        name = str(row["名称"])
        try:
            df = fetch_history(code, start_str, end_str, adjust=ADJUST)
            if df.empty or df.shape[0] < MIN_BARS_PER_SYMBOL:
                print(f"[{i+1}/{len(codes_df)}] {code} {name}: 数据不足，跳过")
                time.sleep(SLEEP_SEC_EACH); continue

            ev = scan_symbol_for_vcp_breakouts(df)
            if not ev.empty:
                ev.insert(0, "code", code)
                ev.insert(1, "name", name)
                all_events.append(ev)

                last_dt = ev["break_date"].max()
                print(f"[{i+1}/{len(codes_df)}] {code} {name}: 触发 {len(ev)} 次，最近一次 {last_dt.date()}")
            else:
                print(f"[{i+1}/{len(codes_df)}] {code} {name}: 无触发")
        except Exception as e:
            print(f"[{i+1}/{len(codes_df)}] {code} {name}: ERROR {e}")
        time.sleep(SLEEP_SEC_EACH)

    events_df = pd.concat(all_events, ignore_index=True)

    # 汇总
    grp = events_df.groupby(["code", "name"], as_index=False).agg(
        trigger_count=("break_date", "count"),
        first_trigger=("break_date", "min"),
        last_trigger =("break_date", "max")
    ).sort_values(["last_trigger", "trigger_count"], ascending=[False, False])

    # 保存
    events_df.to_csv(OUT_EVENTS_CSV, index=False, encoding="utf-8-sig")
    grp.to_csv(OUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    print(f"\n扫描完成，用时 {time.time()-t0:.1f}s")
    print(f"事件明细  → {os.path.abspath(OUT_EVENTS_CSV)}")
    print(f"结果汇总  → {os.path.abspath(OUT_SUMMARY_CSV)}")

    return grp, events_df


# ================== 运行 ==================
if __name__ == "__main__":
    summary, events = scan_a_share_vcp_breakouts()
    if summary is not None and not summary.empty:
        print("\n=== Top 10（最近触发优先）===")
        print(summary.head(10))
