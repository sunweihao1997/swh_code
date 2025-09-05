'''
2025-9-1
This script is to calculate and detecting the ATR shrinking pattern for stocks.
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import sys
import os
import time

# ============= Calculate ATR =============
import pandas_ta as ta

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保索引为 DatetimeIndex 且已排序。"""
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'Date' in df.columns:
            df = df.set_index(pd.to_datetime(df['Date']))
        else:
            raise ValueError("DataFrame 需要 DatetimeIndex 或包含 'Date' 列")
    return df.sort_index()

def atr_ta(df: pd.DataFrame, n: int = 20) -> pd.Series:
    """
    用 pandas_ta 计算 ATR(n)。返回一个与 df 对齐的 Series。
    默认列名为 'ATR_{n}'（pandas_ta 会生成 atr_{n} 的列名，我们重命名一下以统一风格）。
    """
    out = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=n)
    # pandas_ta 返回的列名通常是 'ATR_{n}'
    if isinstance(out, pd.Series):
        return out.rename(f'ATR{n}')
    # 兼容老版本返回DataFrame
    return out.iloc[:, 0].rename(f'ATR{n}')

# ============= Calculate BBW Shrinking Pattern =============
def bb_width_ta(df: pd.DataFrame, n: int = 20, std: float = 2.0) -> pd.Series:
    """
    用 pandas_ta 计算布林带，并取带宽列 BBB（Bandwidth）。
    pandas_ta.bbands 返回列包括：
      BBL_{n}_{std}, BBM_{n}_{std}, BBU_{n}_{std}, BBB_{n}_{std}, BBP_{n}_{std}
    其中 BBB 为带宽。
    """
    bb = ta.bbands(close=df['Close'], length=n, std=std)
    # 兼容不同版本的列名：优先找 'BBB_*'
    if bb is None or bb.empty:
        raise RuntimeError("pandas_ta.bbands 计算失败，返回空结果")
    bw_cols = [c for c in bb.columns if c.startswith('BBB_')]
    if not bw_cols:
        raise RuntimeError("未在 pandas_ta.bbands 结果中找到 BBB 带宽列")
    s = bb[bw_cols[0]]
    return s.rename(f'BBW{n}')

def percentile_rank_last(series: pd.Series, window: int) -> pd.Series:
    """
    返回“当前值在过去 window 个样本中的分位（含当前）”，范围[0,1]。
    用 rolling.apply 计算，不依赖未来数据。
    """
    def _rank(x):
        s = pd.Series(x).dropna()
        if len(s) == 0:
            return np.nan
        # 最后一个点的百分位排名
        return s.rank(pct=True).iloc[-1]
    # 给一个合理的最小样本，避免前期大量 NaN
    return series.rolling(window, min_periods=max(5, window // 5)).apply(_rank, raw=False)

# =============== Weekly Resample ===============

def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """
    以周五为收盘重采样成周线。
    """
    df = ensure_datetime_index(df)
    agg = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    return df.resample('W-FRI').agg(agg).dropna(how='any')


# ============= 日/周低波动标记（用 pandas_ta 指标） =============

def low_volatility_flags(df_daily: pd.DataFrame,
                         d_len: int = 20, w_len: int = 20,
                         d_lookback_days: int = 252,   # ≈12个月
                         w_lookback_weeks: int = 52,   # ≈12个月
                         d_pct: float = 0.20, w_pct: float = 0.35,
                         bb_std: float = 2.0) -> dict:
    """
    计算日/周两套低波标签：
      daily_low_vol: min(rank(ATR20_d), rank(BBW20_d)) <= d_pct
      weekly_low_vol: min(rank(ATR20_w), rank(BBW20_w)) <= w_pct
    返回 {'daily': ddf, 'weekly': wdf}
    """
    df_daily = ensure_datetime_index(df_daily).copy()

    # --- 日线 ---
    d = df_daily.copy()
    d[f'ATR{d_len}_d'] = atr_ta(d, d_len)
    d[f'BBW{d_len}_d'] = bb_width_ta(d, d_len, std=bb_std)
    d['rank_ATR_d'] = percentile_rank_last(d[f'ATR{d_len}_d'], d_lookback_days)
    d['rank_BBW_d'] = percentile_rank_last(d[f'BBW{d_len}_d'], d_lookback_days)
    d['daily_low_vol'] = (d[['rank_ATR_d', 'rank_BBW_d']].min(axis=1) <= d_pct)

    # --- 周线 ---
    w = to_weekly(df_daily)
    w[f'ATR{w_len}_w'] = atr_ta(w, w_len)
    w[f'BBW{w_len}_w'] = bb_width_ta(w, w_len, std=bb_std)
    w['rank_ATR_w'] = percentile_rank_last(w[f'ATR{w_len}_w'], w_lookback_weeks)
    w['rank_BBW_w'] = percentile_rank_last(w[f'BBW{w_len}_w'], w_lookback_weeks)
    w['weekly_low_vol'] = (w[['rank_ATR_w', 'rank_BBW_w']].min(axis=1) <= w_pct)

    return {'daily': d, 'weekly': w}

# ============= 周线背景 + 日线执行 联动 =============
def link_week_daily(daily_df: pd.DataFrame,
                    flags_d: pd.DataFrame,
                    flags_w: pd.DataFrame,
                    exec_window_days: int = 20,
                    start_next_day: bool = True) -> pd.Series:
    """
    当某一周周线满足 weekly_low_vol 后，在后续 exec_window_days 内，
    日线出现 daily_low_vol 的日期标为候选。
    - start_next_day=True 时，窗口从“该周判定日的下一根日K”开始（推荐，避免前视）。
    """
    daily_df = ensure_datetime_index(daily_df)
    w = flags_w['weekly_low_vol'].copy()

    is_window_open = pd.Series(False, index=daily_df.index)

    for dt, cond in w.items():
        if bool(cond):
            # 找到 dt 在日线索引中的位置
            idx = daily_df.index.searchsorted(dt)
            if idx == len(daily_df):
                continue
            # 如果要求从下一根K开始，就 idx+1
            if start_next_day:
                idx = min(idx + 1, len(daily_df) - 1)
            end_idx = min(idx + exec_window_days - 1, len(daily_df) - 1)
            start = daily_df.index[idx]
            end = daily_df.index[end_idx]
            is_window_open.loc[(is_window_open.index >= start) & (is_window_open.index <= end)] = True

    candidates = is_window_open & flags_d['daily_low_vol'].reindex(daily_df.index).fillna(False)
    return candidates

