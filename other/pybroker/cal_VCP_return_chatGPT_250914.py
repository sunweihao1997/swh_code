# -*- coding: utf-8 -*-
"""
VCP Screener Pro — 可直接运行的一体化脚本（精简工程化版）

改进点（相对你的原脚本）：
1) 稳健性：数据抓取加入重试、超时兜底，本地 Parquet 缓存（默认 24h 失效）。
2) 限速：简单令牌桶限速，避免硬 sleep；支持并发度=1（可改）。
3) 交易日/K线计数：用索引位置差计算“最少 K 根”，避免自然日偏差。
4) 信号更严：突破需【收盘价】> 枢轴 & 量能阈值支持分位数或倍数（默认 1.8×SMA50）。
5) 回测更稳：出场日按“下一个可交易日”寻找；可配置止损/止盈/最长持有。
6) CLI/配置化：参数可命令行覆盖；默认跑单票，支持 --all-a-shares 全市场扫描。
7) 可视化：标注自适应大小；保存图与 CSV。

依赖：
  pip install akshare pandas pandas_ta numpy mplfinance matplotlib pyyaml

快速开始：
  python vcp_screener_pro.py --symbol 002594 --start 20230101 --end 20251231
  # 或全市场（耗时长）：
  python vcp_screener_pro.py --all-a-shares --start 20230101 --end 20251231
"""
import os
import io
import time
import math
import json
import yaml
import random
import logging
import hashlib
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Iterable

import numpy as np
import pandas as pd
import pandas_ta as ta
import akshare as ak
import mplfinance as mpf
import matplotlib.pyplot as plt

# =============================
# 日志设置
# =============================
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("VCP")

# =============================
# 配置结构
# =============================
@dataclass
class Config:
    start_date: str = "20230101"
    end_date: str = "20251231"
    exec_window_days: int = 40
    zigzag_threshold: float = 0.05    # 5%
    contraction_min_drop: float = 0.03
    contraction_max_drop: float = 0.40
    contraction_min_bars: int = 5
    need_contractions: int = 2
    volume_break_k: float = 1.8       # 倍数阈值（与 volume_break_q 二选一生效）
    volume_break_q: Optional[float] = None  # 成交量分位阈值（0-1），若给出优先生效
    breakout_look_ahead: int = 20
    wick_ratio_max: float = 0.6       # 上影线占日振幅比例上限（避免假突破），None 则不启用
    stop_loss_pct: float = -0.08
    take_profit_pct: float = 0.25
    max_hold_days: int = 120
    cache_dir: str = ".cache_vcp"
    cache_ttl_hours: int = 24
    charts_dir: str = "/home/sun/paint/vcp_charts"
    results_csv: str = "/home/sun/paint/vcp_breakout_results.csv"
    limit_qps: float = 0.2             # 每秒 0.2 个请求（约 1 次/5 秒）
    concurrent: int = 1                # 预留并发控制（此版逻辑为串行）

# =============================
# 简单令牌桶限速
# =============================
class RateLimiter:
    def __init__(self, qps: float):
        self.capacity = max(1.0, qps * 2)
        self.tokens = self.capacity
        self.refill_rate = qps
        self.last_time = time.time()

    def acquire(self):
        now = time.time()
        delta = now - self.last_time
        self.last_time = now
        self.tokens = min(self.capacity, self.tokens + delta * self.refill_rate)
        if self.tokens < 1.0:
            need = (1.0 - self.tokens) / self.refill_rate
            time.sleep(max(0.0, need))
            self.tokens = 0.0
        else:
            self.tokens -= 1.0

# =============================
# 工具函数
# =============================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def cache_path_for(cfg: Config, key: str) -> str:
    ensure_dir(cfg.cache_dir)
    h = hashlib.md5(key.encode('utf-8')).hexdigest()
    return os.path.join(cfg.cache_dir, f"{h}.parquet")


def save_parquet_safe(df: pd.DataFrame, path: str):
    try:
        df.to_parquet(path, index=True)
    except Exception:
        # 某些环境无 pyarrow/fastparquet，可退化为 csv
        csv_fallback = path.replace('.parquet', '.csv')
        df.to_csv(csv_fallback, encoding='utf-8-sig')
        logger.warning(f"保存 Parquet 失败，已退化为 CSV: {csv_fallback}")


def load_cache(cfg: Config, key: str) -> Optional[pd.DataFrame]:
    p = cache_path_for(cfg, key)
    if not os.path.exists(p):
        # 兼容 CSV 回退
        p_csv = p.replace('.parquet', '.csv')
        if not os.path.exists(p_csv):
            return None
        else:
            mtime = os.path.getmtime(p_csv)
            if time.time() - mtime > cfg.cache_ttl_hours * 3600:
                return None
            try:
                df = pd.read_csv(p_csv)
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                return df
            except Exception:
                return None

    mtime = os.path.getmtime(p)
    if time.time() - mtime > cfg.cache_ttl_hours * 3600:
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


def percentile_rank_last(series: pd.Series, window: int) -> pd.Series:
    def _rank(x: Iterable[float]):
        s = pd.Series(x).dropna()
        if len(s) == 0:
            return np.nan
        return s.rank(pct=True).iloc[-1]
    return series.rolling(window, min_periods=max(5, window // 5)).apply(_rank, raw=False)


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample('W-FRI').agg(agg).dropna(how='any')

# =============================
# 数据获取（带重试 + 缓存 + A/HK 双试）
# =============================

def get_stock_data(cfg: Config, limiter: RateLimiter, stock_code: str,
                   period: str = "daily") -> Optional[pd.DataFrame]:
    key = json.dumps({
        'code': stock_code,
        'period': period,
        'start': cfg.start_date,
        'end': cfg.end_date
    }, ensure_ascii=False)
    cached = load_cache(cfg, key)
    if cached is not None and len(cached) > 0:
        return cached.sort_index()

    def _standardize(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return None
        col_map = {
            '日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'
        }
        df = df.rename(columns=col_map)
        req = ['date', 'open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in req):
            return None
        df = df[req]
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna()
        return df.sort_index()

    attempts = 3
    out_df = None

    for i in range(attempts):
        try:
            limiter.acquire()
            df_a = ak.stock_zh_a_hist(symbol=stock_code, period=period,
                                       start_date=cfg.start_date, end_date=cfg.end_date, adjust="qfq")
            out_df = _standardize(df_a)
            if out_df is not None and len(out_df) > 0:
                break
        except Exception as e:
            logger.debug(f"A股抓取失败尝试{i+1}/{attempts}: {e}")
        time.sleep(0.5 * (i + 1))

    if out_df is None or out_df.empty:
        for i in range(attempts):
            try:
                limiter.acquire()
                df_hk = ak.stock_hk_hist(symbol=stock_code, period=period,
                                         start_date=cfg.start_date, end_date=cfg.end_date, adjust="qfq")
                out_df = _standardize(df_hk)
                if out_df is not None and len(out_df) > 0:
                    break
            except Exception as e:
                logger.debug(f"港股抓取失败尝试{i+1}/{attempts}: {e}")
            time.sleep(0.5 * (i + 1))

    if out_df is None or out_df.empty:
        return None

    # 写缓存
    save_parquet_safe(out_df, cache_path_for(cfg, key))
    return out_df

# =============================
# 低波动执行窗口
# =============================

def find_execution_windows(df_daily: pd.DataFrame, exec_window_days: int = 40) -> pd.Series:
    d = df_daily.copy()
    d['ATR20_d'] = ta.atr(high=d['high'], low=d['low'], close=d['close'], length=20)
    bb = ta.bbands(close=d['close'], length=20, std=2.0)
    # 兼容不同 pandas_ta 版本列名
    if bb is not None and bb.shape[1] >= 4:
        bbw_col = [c for c in bb.columns if 'bandwidth' in str(c).lower() or str(c).upper().startswith('BBB_')]
        if len(bbw_col) == 0:
            # 退化：用 (upper-lower)/mid 近似
            bbw = (bb.iloc[:, 0] - bb.iloc[:, 2]).abs() / (bb.iloc[:, 1].abs() + 1e-9)
        else:
            bbw = bb[bbw_col[0]]
    else:
        bbw = d['close'].rolling(20).std()
    d['BBW20_d'] = bbw

    d['rank_ATR_d'] = percentile_rank_last(d['ATR20_d'], window=252)
    d['rank_BBW_d'] = percentile_rank_last(d['BBW20_d'], window=252)
    d['daily_low_vol'] = (d[['rank_ATR_d', 'rank_BBW_d']].min(axis=1) <= 0.20)

    w = to_weekly(df_daily)
    w['ATR20_w'] = ta.atr(high=w['high'], low=w['low'], close=w['close'], length=20)
    bb_w = ta.bbands(close=w['close'], length=20, std=2.0)
    if bb_w is not None and bb_w.shape[1] >= 4:
        bbw_col_w = [c for c in bb_w.columns if 'bandwidth' in str(c).lower() or str(c).upper().startswith('BBB_')]
        bbw_w = bb_w[bbw_col_w[0]] if len(bbw_col_w) else (bb_w.iloc[:, 0]-bb_w.iloc[:, 2]).abs()/(bb_w.iloc[:,1].abs()+1e-9)
    else:
        bbw_w = w['close'].rolling(20).std()

    w['rank_ATR_w'] = percentile_rank_last(w['ATR20_w'], window=52)
    w['rank_BBW_w'] = percentile_rank_last(w['BBW20_w'] if 'BBW20_w' in w else bbw_w, window=52)
    w['weekly_low_vol'] = (w[['rank_ATR_w', 'rank_BBW_w']].min(axis=1) <= 0.35)

    is_window_open = pd.Series(False, index=df_daily.index)
    for dt_idx, row in w[w['weekly_low_vol']].iterrows():
        start_date = dt_idx + pd.Timedelta(days=1)
        end_date = start_date + pd.Timedelta(days=exec_window_days)
        is_window_open.loc[start_date:end_date] = True

    candidates = is_window_open & d['daily_low_vol'].reindex(df_daily.index).fillna(False)
    return candidates

# =============================
# VCP 识别（ZigZag → 收缩对 → 判定）
# =============================
ContractionPairs = List[Tuple[pd.Timestamp, float, pd.Timestamp, float, float]]

def find_swings_zigzag(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    price_h, price_l = df['high'], df['low']
    n = len(df)
    swing_high = np.full(n, np.nan)
    swing_low = np.full(n, np.nan)
    direction = 0
    max_price, min_price = price_h.iloc[0], price_l.iloc[0]
    max_i, min_i = 0, 0
    for i in range(1, n):
        hi, lo = price_h.iloc[i], price_l.iloc[i]
        if direction == 0:
            if hi > max_price: max_price, max_i = hi, i
            if lo < min_price: min_price, min_i = lo, i
            if (hi - min_price) / max(1e-12, min_price) >= threshold and min_i < i:
                swing_low[min_i] = min_price
                direction, max_price, max_i = 1, hi, i
            elif (max_price - lo) / max(1e-12, max_price) >= threshold and max_i < i:
                swing_high[max_i] = max_price
                direction, min_price, min_i = -1, lo, i
        elif direction == 1:
            if hi > max_price: max_price, max_i = hi, i
            elif (max_price - lo) / max(1e-12, max_price) >= threshold and max_i < i:
                swing_high[max_i] = max_price
                direction, min_price, min_i = -1, lo, i
        else:
            if lo < min_price: min_price, min_i = lo, i
            elif (hi - min_price) / max(1e-12, min_price) >= threshold and min_i < i:
                swing_low[min_i] = min_price
                direction, max_price, max_i = 1, hi, i
    return pd.DataFrame({'swing_high': swing_high, 'swing_low': swing_low}, index=df.index)

def extract_contractions(df: pd.DataFrame, swings: pd.DataFrame,
                          min_drop: float = 0.03, max_drop: float = 0.40,
                          min_bars: int = 5) -> ContractionPairs:
    pairs: ContractionPairs = []
    sh = swings['swing_high'].dropna()
    sl = swings['swing_low'].dropna()
    for t_hi, v_hi in sh.items():
        # 找第一个位于此高点之后的 swing low
        possible_lows = sl[sl.index > t_hi]
        if not len(possible_lows):
            continue
        t_lo, v_lo = possible_lows.index[0], possible_lows.iloc[0]
        # 保证该 low 没被更近的另一高点覆盖
        prev_highs = sh[sh.index < t_lo]
        if not prev_highs.empty and prev_highs.index[-1] != t_hi:
            continue
        # 用 K 线根数衡量间隔
        try:
            bars = df.index.get_loc(t_lo) - df.index.get_loc(t_hi)
        except KeyError:
            bars = (t_lo - t_hi).days
        if bars < min_bars:
            continue
        drop = (v_hi - v_lo) / max(1e-12, v_hi)
        if min_drop <= drop <= max_drop:
            pairs.append((t_hi, float(v_hi), t_lo, float(v_lo), float(drop)))
    return pairs

def is_valid_vcp_relaxed(pairs: ContractionPairs, need_n: int = 2) -> Optional[Dict]:
    if len(pairs) < need_n:
        return None
    sub = pairs[-need_n:]
    drops = [p[4] for p in sub]
    lows = [p[3] for p in sub]
    for i in range(len(drops) - 1):
        if not (drops[i] > drops[i + 1]):
            return None
    first_low = lows[0]
    for i in range(1, len(lows)):
        if not (lows[i] > first_low):
            return None
    pivot_time = sub[-1][0]
    pivot_price = sub[-1][1]
    return {"valid": True, "pairs": sub, "pivot_time": pivot_time, "pivot_price": pivot_price}

# =============================
# 突破检测
# =============================

def _volume_break_mask(scan_df: pd.DataFrame, full_df: pd.DataFrame,
                       volume_break_k: float,
                       volume_break_q: Optional[float]) -> pd.Series:
    vol_sma50 = full_df['volume'].rolling(50, min_periods=20).mean()
    if volume_break_q is not None:
        # 用滚动分位做阈值（相对更稳健）
        vol_q = full_df['volume'].rolling(200, min_periods=50).quantile(volume_break_q)
        thr = vol_q.reindex(scan_df.index)
        return scan_df['volume'] > thr.fillna(method='ffill')
    thr = (volume_break_k * vol_sma50).reindex(scan_df.index)
    return scan_df['volume'] > thr


def find_breakout(df: pd.DataFrame, pivot_time: pd.Timestamp, pivot_price: float,
                  look_ahead: int = 20, volume_break_k: float = 1.8,
                  volume_break_q: Optional[float] = None,
                  wick_ratio_max: Optional[float] = 0.6) -> Optional[pd.Series]:
    # 扫描窗口（按索引定位）
    if pivot_time not in df.index:
        return None
    start_loc = df.index.get_loc(pivot_time)
    scan = df.iloc[start_loc + 1: start_loc + 1 + look_ahead]
    if scan.empty:
        return None
    # 价格条件：收盘 > 枢轴（更严格于“最高价>枢轴”）
    price_ok = scan['close'] > pivot_price
    # 量能条件
    vol_ok = _volume_break_mask(scan, df, volume_break_k, volume_break_q)
    # 上影线过滤（可选）
    if wick_ratio_max is not None:
        rng = (scan['high'] - scan['low']).replace(0, np.nan)
        upper = (scan['high'] - scan['close']) / rng
        wick_ok = upper <= wick_ratio_max
    else:
        wick_ok = pd.Series(True, index=scan.index)
    mask = price_ok & vol_ok & wick_ok
    hits = scan[mask]
    if not hits.empty:
        return hits.iloc[0]
    return None

# =============================
# 可视化
# =============================

def plot_vcp_analysis(df: pd.DataFrame, vcp_result: Dict, breakout_signal: Optional[pd.Series],
                      stock_label: str, save_path: str):
    ensure_dir(os.path.dirname(save_path) or '.')
    vcp_pairs = vcp_result['pairs']
    pivot_price = vcp_result['pivot_price']
    start_date = vcp_pairs[0][0] - pd.Timedelta(days=30)
    end_date = breakout_signal.name + pd.Timedelta(days=15) if breakout_signal is not None else df.index[-1]
    plot_df = df.loc[start_date:end_date]

    contraction_lines = [ [(p[0], p[1]), (p[2], p[3])] for p in vcp_pairs ]
    pivot_start = vcp_pairs[-1][0]
    pivot_end = breakout_signal.name if breakout_signal is not None else plot_df.index[-1]
    pivot_line = [(pivot_start, pivot_price), (pivot_end, pivot_price)]

    add_plots = []
    if breakout_signal is not None:
        buy_marker_df = pd.DataFrame(index=plot_df.index)
        buy_marker_df['signal'] = np.nan
        buy_marker_df.loc[breakout_signal.name, 'signal'] = float(breakout_signal['low']) * 0.98
        markersize = max(40, min(160, int(120 * (len(plot_df) / 120))))
        add_plots.append(mpf.make_addplot(buy_marker_df['signal'], type='scatter', marker='^', color='green', markersize=markersize))

    fig, axlist = mpf.plot(
        plot_df, type='candle', style='yahoo', title=f'VCP: {stock_label}',
        ylabel='Price', volume=True, ylabel_lower='Volume', figsize=(14, 8),
        alines=dict(alines=contraction_lines + [pivot_line], colors=['blue']*len(contraction_lines)+['red'], linewidths=1.2),
        addplot=add_plots, returnfig=True
    )

    ax = axlist[0]
    for i, p in enumerate(vcp_pairs):
        ax.text(p[2], p[3], f' L{i+1}', va='top', fontsize=8, color='black')

    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# =============================
# 回测
# =============================

def next_idx(df: pd.DataFrame, idx) -> Optional[int]:
    try:
        loc = df.index.get_loc(idx)
        if isinstance(loc, slice):
            loc = loc.start
        return loc + 1 if loc + 1 < len(df.index) else None
    except Exception:
        return None


def run_backtest(cfg: Config, df: pd.DataFrame, breakout_signal: pd.Series) -> Dict:
    res: Dict[str, object] = {}
    breakout_date = breakout_signal.name
    entry_price = float(breakout_signal['close'])

    # 定期持有
    for name, days in {'20d_ret': 20, '60d_ret': 60}.items():
        try:
            start_loc = df.index.get_loc(breakout_date)
            exit_loc = start_loc + days
            if exit_loc < len(df.index):
                exit_price = float(df.iloc[exit_loc]['close'])
                ret = (exit_price - entry_price) / entry_price
                res[name] = f"{ret:.2%}"
            else:
                res[name] = "N/A"
        except Exception:
            res[name] = "N/A"

    # 策略持有
    stop_loss_price = entry_price * (1 + cfg.stop_loss_pct)
    take_profit_price = entry_price * (1 + cfg.take_profit_pct)

    start_loc = df.index.get_loc(breakout_date)
    trade_df = df.iloc[start_loc + 1: start_loc + 1 + cfg.max_hold_days]

    exit_reason = "Max Hold"
    exit_price = float(trade_df.iloc[-1]['close']) if not trade_df.empty else entry_price
    days_held = len(trade_df)

    for i, (idx, row) in enumerate(trade_df.iterrows(), start=1):
        low, high = float(row['low']), float(row['high'])
        if low <= stop_loss_price:
            exit_reason = "Stop-Loss"
            exit_price = stop_loss_price
            days_held = i
            break
        if high >= take_profit_price:
            exit_reason = "Take-Profit"
            exit_price = take_profit_price
            days_held = i
            break

    strategy_ret = (exit_price - entry_price) / entry_price
    res['strat_ret'] = f"{strategy_ret:.2%}"
    res['strat_exit_reason'] = exit_reason
    res['strat_days_held'] = days_held
    return res

# =============================
# 单票分析
# =============================

def analyze_single(cfg: Config, limiter: RateLimiter, symbol: str) -> Optional[Dict]:
    df = get_stock_data(cfg, limiter, symbol)
    if df is None or len(df) < 200:
        logger.warning(f"{symbol}: 数据不足或抓取失败")
        return None

    candidates = find_execution_windows(df, cfg.exec_window_days)
    if not candidates.any():
        logger.info(f"{symbol}: 未找到低波动执行窗口")
        return None

    last_candidate_date = candidates[candidates].index[-1]
    # 从最后候选日前 140 根 K 起做形态分析
    last_loc = df.index.get_loc(last_candidate_date)
    start_loc = max(0, last_loc - 140)
    analysis_df = df.iloc[start_loc:last_loc + 1]

    swings = find_swings_zigzag(analysis_df, threshold=cfg.zigzag_threshold)
    contractions = extract_contractions(analysis_df, swings,
                                        min_drop=cfg.contraction_min_drop,
                                        max_drop=cfg.contraction_max_drop,
                                        min_bars=cfg.contraction_min_bars)
    vcp_result = is_valid_vcp_relaxed(contractions, need_n=cfg.need_contractions)
    if not vcp_result:
        logger.info(f"{symbol}: 无有效 VCP 形态")
        return None

    breakout = find_breakout(df,
                             vcp_result['pivot_time'],
                             vcp_result['pivot_price'],
                             look_ahead=cfg.breakout_look_ahead,
                             volume_break_k=cfg.volume_break_k,
                             volume_break_q=cfg.volume_break_q,
                             wick_ratio_max=cfg.wick_ratio_max)
    if breakout is None:
        logger.info(f"{symbol}: 有 VCP 但尚未突破")
        return None

    backtest = run_backtest(cfg, df, breakout)

    # 保存图
    safe_symbol = ''.join(ch for ch in symbol if ch.isalnum())
    breakout_date_str = breakout.name.strftime('%Y-%m-%d')
    ensure_dir(cfg.charts_dir)
    chart_path = os.path.join(cfg.charts_dir, f"{safe_symbol}_{breakout_date_str}.png")
    plot_vcp_analysis(df, vcp_result, breakout, f"{symbol}", chart_path)

    result = {
        'Stock Code': symbol,
        'Pivot Time': vcp_result['pivot_time'].date(),
        'Pivot Price': f"{vcp_result['pivot_price']:.2f}",
        'Breakout Date': breakout.name.date(),
        'Breakout Close': f"{float(breakout['close']):.2f}",
        'Contraction Depths': ', '.join([f"{p[4]*100:.1f}%" for p in vcp_result['pairs']]),
        **backtest
    }
    logger.info(f"{symbol}: 成功，突破于 {result['Breakout Date']}，图已保存 {chart_path}")
    return result

# =============================
# 全市场扫描（A股）
# =============================

def screen_all_a_shares(cfg: Config, limiter: RateLimiter) -> List[Dict]:
    try:
        limiter.acquire()
        stock_list = ak.stock_zh_a_spot_em()
    except Exception as e:
        logger.error(f"获取 A 股列表失败: {e}")
        return []
    results: List[Dict] = []
    total = len(stock_list)
    logger.info(f"开始扫描 A 股 {total} 只……（可能耗时较长）")
    for i, row in stock_list.iterrows():
        code = str(row.get('代码', '')).strip()
        name = str(row.get('名称', '')).strip()
        label = f"{code} {name}" if name else code
        logger.info(f"[{i+1}/{total}] 分析 {label}")
        try:
            r = analyze_single(cfg, limiter, code)
            if r:
                r['Stock Name'] = name
                results.append(r)
        except Exception as e:
            logger.warning(f"{label}: 处理异常 {e}")
    return results

# =============================
# 结果落盘
# =============================

def save_results(cfg: Config, results: List[Dict]):
    if not results:
        logger.info("没有命中结果，跳过保存。")
        return
    df = pd.DataFrame(results)
    # 计算统计（将百分比列转为数值）
    for col in ['20d_ret', '60d_ret', 'strat_ret']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].str.rstrip('%'), errors='coerce') / 100.0
    logger.info(
        "回测汇总：样本=%d，20日均值=%.2f%%，60日均值=%.2f%%，策略均值=%.2f%%，胜率=%.2f%%" % (
            len(df),
            100 * df['20d_ret'].mean(skipna=True),
            100 * df['60d_ret'].mean(skipna=True),
            100 * df['strat_ret'].mean(skipna=True),
            100 * (df['strat_ret'] > 0).mean()
        )
    )
    # 导出 CSV（将百分比转回字符串）
    for col in ['20d_ret', '60d_ret', 'strat_ret']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    df.to_csv(cfg.results_csv, index=False, encoding='utf-8-sig')
    logger.info(f"结果已保存至 {cfg.results_csv}")

# =============================
# CLI
# =============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VCP Screener Pro")
    parser.add_argument('--symbol', type=str, default='002594', help='单票代码（默认 002594）')
    parser.add_argument('--start', type=str, default=None, help='开始日期 YYYYMMDD')
    parser.add_argument('--end', type=str, default=None, help='结束日期 YYYYMMDD')
    parser.add_argument('--all-a-shares', action='store_true', help='扫描全 A 股')
    parser.add_argument('--cfg', type=str, default=None, help='YAML 配置文件路径')
    args = parser.parse_args()

    cfg = Config()
    if args.cfg and os.path.exists(args.cfg):
        with open(args.cfg, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    if args.start:
        cfg.start_date = args.start
    if args.end:
        cfg.end_date = args.end

    ensure_dir(cfg.charts_dir)
    limiter = RateLimiter(cfg.limit_qps)

    if args.all_a_shares:
        results = screen_all_a_shares(cfg, limiter)
        save_results(cfg, results)
    else:
        out = analyze_single(cfg, limiter, args.symbol)
        if out:
            save_results(cfg, [out])
        else:
            logger.info("单票未命中或数据不足。")
