'''
2025-9-4
This script is to calculate and detecting the stock volume energy break pattern.
'''
import numpy as np
import pandas as pd

# ---------- 周线重采样 ----------
def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}
    return df.resample('W-FRI').agg(agg).dropna(how='any')

# ---------- 单日量能（sma/ema/percentile 三模式） ----------
def daily_breakout_volume_ok(df: pd.DataFrame,
                             day: pd.Timestamp,
                             ma_len: int = 50,
                             k_mult: float = 1.8,
                             mode: str = "sma",              # "sma" | "ema" | "percentile"
                             pct_lookback: int = 252,
                             pct_thresh: float = 0.80) -> dict:
    if day not in df.index:
        return {'ok': False, 'reason': 'day_not_in_df', 'date': day}

    hist = df.loc[:day].copy()
    vol_today = float(hist.at[day, 'Volume'])

    if mode == "sma":
        ref = hist['Volume'].rolling(ma_len, min_periods=max(10, ma_len//2)).mean().iat[-1]
        ratio = vol_today / ref if (pd.notna(ref) and ref > 0) else np.nan
        ok = (not np.isnan(ratio)) and (ratio >= k_mult)
        return {'ok': bool(ok), 'vol': vol_today, 'ref': float(ref) if pd.notna(ref) else np.nan,
                'ratio': ratio, 'pct': None, 'reason': None if ok else 'ratio_lt_k', 'date': day}

    elif mode == "ema":
        ref = hist['Volume'].ewm(span=ma_len, adjust=False, min_periods=max(10, ma_len//2)).mean().iat[-1]
        ratio = vol_today / ref if (pd.notna(ref) and ref > 0) else np.nan
        ok = (not np.isnan(ratio)) and (ratio >= k_mult)
        return {'ok': bool(ok), 'vol': vol_today, 'ref': float(ref) if pd.notna(ref) else np.nan,
                'ratio': ratio, 'pct': None, 'reason': None if ok else 'ratio_lt_k', 'date': day}

    elif mode == "percentile":
        win = hist['Volume'].tail(pct_lookback)
        if win.empty or len(win.dropna()) < max(20, pct_lookback//5):
            return {'ok': False, 'vol': vol_today, 'ref': np.nan,
                    'ratio': None, 'pct': None, 'reason': 'insufficient_history', 'date': day}
        pct = win.rank(pct=True).iloc[-1]
        ok = pct >= pct_thresh
        return {'ok': bool(ok), 'vol': vol_today, 'ref': np.nan,
                'ratio': None, 'pct': float(pct), 'reason': None if ok else 'pct_lt_thresh', 'date': day}

    else:
        raise ValueError("mode 必须是 'sma' | 'ema' | 'percentile'")

# ---------- 周线放量确认（倍数 or 分位） ----------
def weekly_breakout_volume_ok(
    df: pd.DataFrame,
    breakout_date: pd.Timestamp,
    ma_weeks: int = 26,
    k_mult_week: float = 1.5,
    use_percentile: bool = True,
    pct_thresh: float = 0.80,
    _weekly_cache: dict | None = None,
    confirm_timing: str = "week_close",  # "week_close" | "partial_week"
) -> dict:
    """
    confirm_timing:
      - "week_close": 仅在周线收盘（周五）用整周量确认，周中返回 {'ok': False, 'reason': 'await_week_close'}
      - "partial_week": 用当周截至 breakout_date 的“已发生周量”做近似确认（更灵敏，口径有差异）
    """
    # 周线缓存
    if _weekly_cache is not None and 'w' in _weekly_cache:
        w = _weekly_cache['w']
    else:
        w = to_weekly(df)
        if _weekly_cache is not None:
            _weekly_cache['w'] = w

    # 找到该日所在的周末
    pos = w.index.searchsorted(breakout_date)
    if pos >= len(w.index):
        return {'ok': False, 'reason': 'break_after_last_week'}

    wk_end = w.index[pos]
    wk_start = w.index[pos-1] + pd.Timedelta(days=1) if pos > 0 else (wk_end - pd.Timedelta(days=6))

    # 计算周量均线
    if _weekly_cache is not None and 'VolMA' in _weekly_cache:
        w_volma = _weekly_cache['VolMA']
        w = w.copy(); w['VolMA'] = w_volma
    else:
        w = w.copy()
        w['VolMA'] = w['Volume'].rolling(ma_weeks, min_periods=max(6, ma_weeks//3)).mean()
        if _weekly_cache is not None:
            _weekly_cache['VolMA'] = w['VolMA']

    # —— 两种确认口径 ——
    if confirm_timing == "week_close":
        # 周中不确认，避免前视
        if breakout_date < wk_end:
            return {'ok': False, 'reason': 'await_week_close', 'week_end': wk_end}
        # 周五收盘后用完整周量
        wvol = float(w.at[wk_end, 'Volume'])
    elif confirm_timing == "partial_week":
        # 用当周截至当天的“部分周量”
        mask = (df.index >= wk_start) & (df.index <= breakout_date)
        wvol = float(df.loc[mask, 'Volume'].sum())
    else:
        raise ValueError("confirm_timing must be 'week_close' or 'partial_week'")

    # 参考均量与分位
    wv_ma = float(w.at[wk_end, 'VolMA'])
    ratio = wvol / wv_ma if (pd.notna(wv_ma) and wv_ma > 0) else np.nan

    # 当前周量（完整或部分）在“近 ma_weeks 周完整周量”里的分位（近似；更严谨可做“部分周对比部分周”的分布）
    hist = w['Volume'].iloc[max(0, w.index.get_loc(wk_end) - ma_weeks + 1): w.index.get_loc(wk_end) + 1]
    pct_rank = (pd.Series(hist.values).rank(pct=True).iloc[-1]) if len(hist) > 0 else np.nan

    if use_percentile:
        ok = (not np.isnan(pct_rank)) and (pct_rank >= pct_thresh)
    else:
        ok = (not np.isnan(ratio)) and (ratio >= k_mult_week)

    return {
        'ok': bool(ok),
        'week_end': wk_end,
        'wvol': wvol, 'wvol_ma': wv_ma, 'ratio': ratio, 'pct_rank': pct_rank,
        'confirm_timing': confirm_timing
    }


# ---------- 统一入口：价 + 日量能 + （可选）周线放量 ----------
def find_breakout_signals_v3(df: pd.DataFrame,
                             pivot_time: pd.Timestamp,
                             pivot_price: float,
                             buf: float = 0.01,
                             look_ahead_days: int = 20,
                             # —— 日线量能参数（透传给 daily_breakout_volume_ok）
                             vol_mode: str = "sma",          # "sma" | "ema" | "percentile"
                             vol_ma_len: int = 50,
                             vol_k_mult: float = 1.8,
                             vol_pct_lookback: int = 252,
                             vol_pct_thresh: float = 0.80,
                             # —— 周线放量确认（新增）
                             weekly_required: bool = True,   # True=必须通过周线确认；False=仅日线即可
                             weekly_ma_weeks: int = 26,
                             weekly_k_mult: float = 1.5,
                             weekly_use_percentile: bool = True,
                             weekly_confirm_timing="week_close",
                             weekly_pct_thresh: float = 0.80) -> pd.DataFrame:
    """
    返回仅包含“价 + 日量能 + （可选）周线放量”均通过的突破日。
    输出列：
      ['Close','Volume','price_ok','vol_ok','vol_ratio','vol_ref','vol_pct',
       'weekly_ok','weekly_ratio','weekly_pct','pivot_price']
    """
    # --- pivot 后扫描区间 ---
    if pivot_time in df.index:
        start_idx = df.index.get_loc(pivot_time)
    else:
        start_idx = df.index.searchsorted(pivot_time)
    start_idx = min(start_idx + 1, len(df) - 1)
    end_idx   = min(start_idx + look_ahead_days, len(df) - 1)
    if start_idx >= end_idx:
        return pd.DataFrame()

    seg = df.iloc[start_idx:end_idx+1].copy()

    # --- 价格条件 ---
    seg['price_ok'] = seg['Close'] >= pivot_price * (1.0 + buf)

    # --- 日线量能条件 ---
    vol_ok_list, vol_ratio_list, vol_ref_list, vol_pct_list = [], [], [], []
    for dt in seg.index:
        res = daily_breakout_volume_ok(
            df, dt,
            ma_len=vol_ma_len, k_mult=vol_k_mult,
            mode=vol_mode, pct_lookback=vol_pct_lookback, pct_thresh=vol_pct_thresh
        )
        vol_ok_list.append(bool(res.get('ok', False)))
        vol_ratio_list.append(res.get('ratio', np.nan))
        vol_ref_list.append(res.get('ref', np.nan))
        vol_pct_list.append(res.get('pct', np.nan))

    seg['vol_ok']    = vol_ok_list
    seg['vol_ratio'] = vol_ratio_list
    seg['vol_ref']   = vol_ref_list
    seg['vol_pct']   = vol_pct_list

    # --- 周线放量确认（可选，默认必须） ---
    weekly_ok_list, weekly_ratio_list, weekly_pct_list = [], [], []
    weekly_cache = {}  # 避免重复计算周线
    for dt in seg.index:
        if not weekly_required:
            weekly_ok_list.append(True); weekly_ratio_list.append(np.nan); weekly_pct_list.append(np.nan)
            continue
        wres = weekly_breakout_volume_ok(
            df, dt,
            ma_weeks=weekly_ma_weeks,
            k_mult_week=weekly_k_mult,
            use_percentile=weekly_use_percentile,
            pct_thresh=weekly_pct_thresh,
            _weekly_cache=weekly_cache
        )
        weekly_ok_list.append(bool(wres.get('ok', False)))
        weekly_ratio_list.append(wres.get('ratio', np.nan))
        weekly_pct_list.append(wres.get('pct_rank', np.nan))

    seg['weekly_ok']    = weekly_ok_list
    seg['weekly_ratio'] = weekly_ratio_list
    seg['weekly_pct']   = weekly_pct_list

    # --- 综合条件 ---
    if weekly_required:
        seg['breakout'] = seg['price_ok'] & seg['vol_ok'] & seg['weekly_ok']
    else:
        seg['breakout'] = seg['price_ok'] & seg['vol_ok']

    seg['pivot_price'] = float(pivot_price)

    return seg.loc[seg['breakout'],
                   ['Close','Volume','price_ok','vol_ok','vol_ratio','vol_ref','vol_pct',
                    'weekly_ok','weekly_ratio','weekly_pct','pivot_price']]
