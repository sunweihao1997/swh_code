'''
20250902
This script is to calculate the ACP shrinking pattern for stocks
'''
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union, Dict

# --- 1) 摆动点识别：中心滑窗极值 ---
def find_swings(df, window=5):
    """
    用 center=True 的 rolling 极值标记 swing high/low。
    window 越大，越“抗噪”；越小，越灵敏。
    """
    hi, lo = df['High'], df['Low']
    is_high = hi.eq(hi.rolling(window=window, center=True).max())
    is_low  = lo.eq(lo.rolling(window=window, center=True).min())
    swings = pd.DataFrame({
        'swing_high': np.where(is_high, hi, np.nan),
        'swing_low' : np.where(is_low,  lo, np.nan),
    }, index=df.index)
    return swings


# --- 2) 提取回撤序列：high → 后继 low ---
def extract_contractions(df, swings, lookback=90, min_drop=0.03, max_drop=0.40):
    """
    在最近 lookback 根K线内，寻找每个 swing high 后面遇到的第一个 swing low，
    计算回撤比例 drop = (high - low)/high；过滤过小/过大的回撤。
    返回 [(hi_idx, hi_val, lo_idx, lo_val, drop), ...]
    """
    d = df.iloc[-lookback:].copy()
    sw = swings.loc[d.index]
    seq = []
    pending_high = None  # (idx, value)

    for i, t in enumerate(d.index):
        hi = sw.at[t, 'swing_high']
        lo = sw.at[t, 'swing_low']
        if not np.isnan(hi):
            pending_high = (t, float(hi))
        # 碰到 low 时，如前面存在未匹配的 high，则计算 drop
        if (pending_high is not None) and (not np.isnan(lo)):
            hi_t, hi_v = pending_high
            drop = (hi_v - float(lo)) / hi_v if hi_v > 0 else np.nan
            if pd.notna(drop) and (min_drop <= drop <= max_drop):
                seq.append((hi_t, hi_v, t, float(lo), float(drop)))
            pending_high = None

    return seq

# --- 3) advanced 提取回撤序列 ---
def extract_contractions_v2(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    lookback: int = 90,
    min_drop: float = 0.03,
    max_drop: float = 0.40,
    min_bars_per_leg: int = 5,
    max_last_drop: Optional[float] = None,
) -> Dict[str, object]:
    """
    在最近 lookback 根K线上，顺序配对“最近一个 swing high → 后继出现的第一个 swing low”，
    计算回撤比例 drop = (high - low)/high，并做以下抗噪过滤：
      1) 回撤幅度范围：min_drop ≤ drop ≤ max_drop
      2) 段间最少K线：两个摆动点之间至少 min_bars_per_leg 根K线
      3) 末段更紧（可选）：若 max_last_drop 给定，则要求最后一段 drop ≤ max_last_drop

    参数
    ----
    df : DataFrame
        原始K线（索引为时间；至少需要 High/Low/Close 用于对齐索引）
    swings : DataFrame
        与 df 同索引，包含列：
          - 'swing_high'：仅在摆动高点处为高点价格，其它位置为 NaN
          - 'swing_low' ：仅在摆动低点处为低点价格，其它位置为 NaN
    lookback : int
        仅在最近 lookback 根K线内寻找收缩段
    min_drop, max_drop : float
        单段回撤的幅度过滤（比例，0~1）
    min_bars_per_leg : int
        高→低这段之间至少需要的K线数量，防止太贴近的锯齿
    max_last_drop : float or None
        若不为 None，则要求“最后一段”的回撤 ≤ 该阈值，用于强调“末端更紧”

    返回
    ----
    result : dict
        {
          'pairs': List[Tuple[Timestamp, float, Timestamp, float, float]],
                    # 每个元素为 (hi_t, hi_v, lo_t, lo_v, drop)
          'last_drop': float or None,   # 最后一段的回撤，若无则 None
          'passed_last_tight': bool,    # 是否满足末端更紧（若未设置 max_last_drop，则恒 True）
        }

    说明
    ----
    - 每个 swing high 只会与其“后继遇到的第一个 swing low”配对一次；
    - 若窗口结尾只剩一个未被配对的 swing high，会被忽略（很合理）；
    - 建议在上游的 swing 检测中使用“居中滑窗极值/ATR自适应ZigZag”等稳健方法。
    """

    # 取最近 lookback 根K线并对齐 swings
    d = df.iloc[-lookback:].copy()
    sw = swings.loc[d.index]

    pairs: List[Tuple[pd.Timestamp, float, pd.Timestamp, float, float]] = []

    pending_high: Optional[Tuple[pd.Timestamp, float]] = None
    last_high_i: Optional[int] = None

    for i, t in enumerate(d.index):
        hi = sw.at[t, 'swing_high'] if 'swing_high' in sw.columns else np.nan
        lo = sw.at[t, 'swing_low']  if 'swing_low'  in sw.columns else np.nan

        # 遇到摆动高点：记录为“待配对”的最近高点
        if pd.notna(hi):
            pending_high = (t, float(hi))
            last_high_i = i

        # 遇到摆动低点，且手里有待配对高点：尝试配对
        if (pending_high is not None) and pd.notna(lo):
            # 1) 段间最少K线过滤
            if (last_high_i is not None) and ((i - last_high_i) < int(min_bars_per_leg)):
                # 太近了，可能是噪音尖点，跳过这次 low（不清空 pending_high，等更合适的low）
                continue

            hi_t, hi_v = pending_high
            lo_v = float(lo)

            # 基本健壮性：高点应该大于低点
            if hi_v <= lo_v:
                # 形态异常，丢弃这次配对并清空（避免重复卡住）
                pending_high = None
                last_high_i = None
                continue

            drop = (hi_v - lo_v) / hi_v if hi_v > 0 else np.nan

            # 2) 回撤幅度区间过滤
            if pd.notna(drop) and (min_drop <= drop <= max_drop):
                pairs.append((hi_t, hi_v, t, lo_v, float(drop)))

            # 不论是否入选，这个 high 已经和“第一个 low”完成了配对
            pending_high = None
            last_high_i = None

    # 3) 末端更紧过滤（仅作标记，不强行删除历史pairs，方便你上游决定如何用）
    last_drop = pairs[-1][4] if pairs else None
    if (pairs) and (max_last_drop is not None):
        passed_last_tight = (last_drop <= max_last_drop)
    else:
        passed_last_tight = True

    return {
        'pairs': pairs,
        'last_drop': last_drop,
        'passed_last_tight': bool(passed_last_tight)
    }

# --- 4) 回撤递减与低点抬高 v1 ---
def is_valid_vcp_sequence(seq, need_n=3, tol=0.0, enforce_higher_lows=True):
    """
    seq: [(hi_t, hi_v, lo_t, lo_v, drop), ...]
    取最近 need_n 次，要求 drop 递减，且 low 递增（可选）
    """
    if len(seq) < need_n:
        return False, None
    sub = seq[-need_n:]  # 取最近 need_n 次
    drops = [x[4] for x in sub]
    if not all(drops[i] > drops[i+1] + tol for i in range(len(drops)-1)):
        return False, None
    if enforce_higher_lows:
        lows = [x[3] for x in sub]
        if not all(lows[i] < lows[i+1] for i in range(len(lows)-1)):
            return False, None
    return True, sub

# --- 5) 回撤递减与低点抬高 v2 ---
from typing import List, Tuple, Optional, Union, Dict
import pandas as pd

PairsT = List[Tuple[pd.Timestamp, float, pd.Timestamp, float, float]]

def is_valid_vcp_sequence_v2(
    seq_or_result: Union[PairsT, Dict[str, object]],
    need_n: int = 3,
    tol_drop: float = 0.0,           # 回撤递减的最小差额容差（如 0.01 = 至少差1pct）
    enforce_higher_lows: bool = True,
    low_tol: float = 0.0,            # 低点抬高的最小价差（如 0.02 元，或用波动自适应）
    require_last_tight: bool = False # 是否强制要求“末端更紧”（依赖 v2 的 max_last_drop 判断）
) -> Dict[str, object]:
    """
    校验 VCP 收缩序列的结构条件：
      1) 最近 need_n 段回撤幅度严格递减：drops[i] > drops[i+1] + tol_drop
      2) （可选）低点抬高：lows[i+1] - lows[i] > low_tol
      3) （可选）强制“末端更紧”：要求 extract_contractions_v2 的 passed_last_tight == True

    参数
    ----
    seq_or_result : 
        - 直接传入 pairs 列表：[(hi_t, hi_v, lo_t, lo_v, drop), ...]
        - 或传入 extract_contractions_v2 的返回 dict（含 'pairs' / 'passed_last_tight' 等）
    need_n : 需要验证的最近段数（典型 3）
    tol_drop : 回撤递减容差
    enforce_higher_lows : 是否要求低点抬高
    low_tol : 低点抬高的最小价差
    require_last_tight : 是否强制要求“末端更紧”通过

    返回
    ----
    dict:
      {
        'valid': bool,                 # 是否通过
        'reason': str or None,         # 未通过原因
        'sub_pairs': PairsT or None,   # 用于校验的最近 need_n 段
        'drops': list or None,         # 对应的回撤序列
        'lows': list or None,          # 对应的低点序列
        'last_drop': float or None,    # 最近一段回撤
        'passed_last_tight': bool,     # v2“末端更紧”是否通过（若未提供则为 True）
        'pivot_time': pd.Timestamp or None, # pivot 候选（最近一段的 swing high 时间）
        'pivot_price': float or None,        # pivot 候选（最近一段的 swing high 价格）
      }
    """
    # 1) 取出 pairs 与“末端更紧”标记
    if isinstance(seq_or_result, dict):
        pairs: PairsT = seq_or_result.get('pairs', [])  # type: ignore
        passed_last_tight = bool(seq_or_result.get('passed_last_tight', True))
        last_drop = seq_or_result.get('last_drop', None)
    else:
        pairs = seq_or_result
        passed_last_tight = True
        last_drop = pairs[-1][4] if pairs else None

    # 2) 基本长度检查
    if len(pairs) < need_n:
        return {
            'valid': False,
            'reason': f'not_enough_pairs(<{need_n})',
            'sub_pairs': None, 'drops': None, 'lows': None,
            'last_drop': last_drop, 'passed_last_tight': passed_last_tight,
            'pivot_time': None, 'pivot_price': None,
        }

    sub = pairs[-need_n:]
    drops = [x[4] for x in sub]
    lows  = [x[3] for x in sub]

    # 3) 回撤递减 + 容差
    for i in range(len(drops) - 1):
        if not (drops[i] > drops[i+1] + tol_drop):
            return {
                'valid': False,
                'reason': f'drops_not_decreasing_at_{i}: {drops[i]:.4f} <= {drops[i+1]:.4f} + {tol_drop:.4f}',
                'sub_pairs': sub, 'drops': drops, 'lows': lows,
                'last_drop': last_drop, 'passed_last_tight': passed_last_tight,
                'pivot_time': None, 'pivot_price': None,
            }

    # 4) 低点抬高 + 容差
    if enforce_higher_lows:
        for i in range(len(lows) - 1):
            if not (lows[i+1] - lows[i] > low_tol):
                return {
                    'valid': False,
                    'reason': f'lows_not_increasing_at_{i}: {lows[i+1]:.4f} - {lows[i]:.4f} <= {low_tol:.4f}',
                    'sub_pairs': sub, 'drops': drops, 'lows': lows,
                    'last_drop': last_drop, 'passed_last_tight': passed_last_tight,
                    'pivot_time': None, 'pivot_price': None,
                }

    # 5) （可选）强制“末端更紧”
    if require_last_tight and (not passed_last_tight):
        return {
            'valid': False,
            'reason': 'last_leg_not_tight_enough',
            'sub_pairs': sub, 'drops': drops, 'lows': lows,
            'last_drop': last_drop, 'passed_last_tight': passed_last_tight,
            'pivot_time': None, 'pivot_price': None,
        }

    # 6) 生成 pivot 候选：最近一段的 swing high（时间与价格）
    pivot_time = sub[-1][0]
    pivot_price = sub[-1][1]

    return {
        'valid': True,
        'reason': None,
        'sub_pairs': sub, 'drops': drops, 'lows': lows,
        'last_drop': last_drop, 'passed_last_tight': passed_last_tight,
        'pivot_time': pivot_time, 'pivot_price': pivot_price,
    }
