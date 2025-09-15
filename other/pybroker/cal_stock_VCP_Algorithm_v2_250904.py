'''
20250902
This script is to calculate the ACP shrinking pattern for stocks

VCP detection with no-lookahead flags
2025-09-04

功能概览
--------
1) 摆动点识别：center=True 的 rolling 极值（更抗噪）
2) 回撤段提取：配对 (swing high -> 后继 swing low)，过滤幅度与最少K数
3) VCP 校验：回撤递减 + 低点抬高（low_tol 支持 abs/pct/atr 三种模式）
4) 无前视封装：统一延迟确认中心窗的未来依赖，输出与日线对齐的 vcp_ok / pivot_price

依赖
----
pandas, numpy
'''
from __future__ import annotations
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import pandas as pd

# ========= 基础工具 =========

def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """确保索引是 DatetimeIndex，去重并排序。"""
    _df = df.copy()
    if not isinstance(_df.index, pd.DatetimeIndex):                    #isinstance 是 Python 内置函数，用来判断一个对象是不是某个类型（或某些类型之一）的实例。
        if "Date" in _df.columns:
            _df["Date"] = pd.to_datetime(_df["Date"], errors="coerce") # 把 _df["Date"] 这一列转换成 pandas 的日期时间类型 (datetime64[ns]) ; errors="coerce" 的意思是如果转换失败就设置为 NaT (Not a Time)，而不是报错。
            _df = _df.dropna(subset=["Date"]).set_index("Date")
        else:
            raise ValueError("DataFrame 需要 DatetimeIndex 或包含 'Date' 列")
    _df = _df[~_df.index.duplicated(keep="last")].sort_index()
    return _df


def atr_wilder(df: pd.DataFrame, n: int = 20) -> pd.Series:
    h, l, c = df["High"].astype(float), df["Low"].astype(float), df["Close"].astype(float)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()  # Wilder's smoothing
    return atr.rename(f"ATR{n}")



# ========= 1) 摆动点识别（中心滑窗） =========

def find_swings(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    用 center=True 的 rolling 极值标记 swing high / swing low。
    注意：center=True 会依赖“未来”K根数据，务必在后续统一延迟 half=window//2 进行确认！
    """
    df = ensure_datetime_index(df)
    hi, lo = df["High"], df["Low"]
    is_high = hi.eq(hi.rolling(window=window, center=True).max())
    is_low  = lo.eq(lo.rolling(window=window, center=True).min())
    swings = pd.DataFrame(
        {"swing_high": np.where(is_high, hi, np.nan),
         "swing_low":  np.where(is_low,  lo, np.nan)},
        index=df.index
    )
    return swings

import numpy as np
import pandas as pd

def find_swings_zigzag(
    df: pd.DataFrame,
    threshold: float = 0.05,
    *,
    mode: str = "percent",           # "percent" 或 "abs"
    use_high_low: bool = True,       # True: 用 High/Low 做极值；False: 仅用 Close
    include_last: bool = False       # 是否把最后一个候选拐点也输出为 swing（未被反向确认）
) -> pd.DataFrame:
    """
    用 ZigZag 规则标记 swing high / swing low（无前视；O(n)）。
    - 当价格自最近极低点向上反弹 ≥ threshold（百分比或点数）时，确认该极低点为 swing low；
    - 当价格自最近极高点向下回撤 ≥ threshold 时，确认该极高点为 swing high。

    参数
    ----
    df : DataFrame，需要包含列：
         - 必需: "Close"
         - 可选: "High", "Low"（当 use_high_low=True 时使用）
    threshold : float
         - 当 mode="percent" 时，表示比例阈值（如 0.05 表示 5%）
         - 当 mode="abs" 时，表示绝对点数阈值（如 2.0 表示 2 元）
    mode : {"percent", "abs"}
    use_high_low : bool
         - True：用 High/Low 作为即时极值（推荐，转折更稳健）
         - False：仅用 Close 价格判定
    include_last : bool
         - True：在序列结束时输出最后一个候选极值为 swing（可能未被反向确认）

    返回
    ----
    swings : DataFrame，索引与输入一致，包含：
        - swing_high: 在被确认的波峰位置给出价格，否则为 NaN
        - swing_low : 在被确认的波谷位置给出价格，否则为 NaN
    """
    # --- 预处理 ---
    if "Close" not in df.columns:
        raise ValueError("DataFrame 必须包含 'Close' 列。")
    price_h = df["High"] if (use_high_low and "High" in df.columns) else df["Close"]
    price_l = df["Low"]  if (use_high_low and "Low"  in df.columns) else df["Close"]
    close = df["Close"]

    n = len(df)
    if n == 0:
        return pd.DataFrame({"swing_high": [], "swing_low": []}, index=df.index)

    idx = df.index

    # 输出容器
    swing_high = np.full(n, np.nan, dtype=float)
    swing_low  = np.full(n, np.nan, dtype=float)

    # 初始状态
    direction = 0  # 0=未定, 1=上涨段（寻找峰）, -1=下跌段（寻找谷）
    # 候选极值（价格与索引）
    max_price = price_h.iloc[0]
    min_price = price_l.iloc[0]
    max_i = 0
    min_i = 0

    # 阈值判断函数
    if mode == "percent":
        def risen_from_min(curr_high, base_min):
            # 从最近极低向上反弹比例 >= threshold ?
            return (curr_high - base_min) / base_min >= threshold
        def pulled_from_max(curr_low, base_max):
            # 从最近极高向下回撤比例 >= threshold ?
            return (base_max - curr_low) / base_max >= threshold
    elif mode == "abs":
        def risen_from_min(curr_high, base_min):
            return (curr_high - base_min) >= threshold
        def pulled_from_max(curr_low, base_max):
            return (base_max - curr_low) >= threshold
    else:
        raise ValueError("mode 只能为 'percent' 或 'abs'。")

    # 单次遍历
    for i in range(1, n):
        hi = price_h.iloc[i]
        lo = price_l.iloc[i]
        cl = close.iloc[i]

        # 尚未建立方向：同时追踪极高与极低，直到某一侧突破阈值
        if direction == 0:
            # 更新候选极值
            if hi > max_price:
                max_price, max_i = hi, i
            if lo < min_price:
                min_price, min_i = lo, i
            # 判断是否形成第一段
            if risen_from_min(hi, min_price) and min_i < i:
                # 从极低反弹足够 → 之前的 min 是 swing low
                swing_low[min_i] = min_price
                direction = 1
                # 上涨段开始后，峰值从当前 hi 起步
                max_price, max_i = hi, i
            elif pulled_from_max(lo, max_price) and max_i < i:
                # 从极高回撤足够 → 之前的 max 是 swing high
                swing_high[max_i] = max_price
                direction = -1
                # 下跌段开始后，谷值从当前 lo 起步
                min_price, min_i = lo, i
            continue

        # 已有方向：上涨段寻找峰，下跌段寻找谷
        if direction == 1:
            # 上涨段持续刷新峰值
            if hi > max_price:
                max_price, max_i = hi, i
            # 若从峰值回撤达到阈值 → 确认该峰为 swing high，方向转空
            if pulled_from_max(lo, max_price) and max_i < i:
                swing_high[max_i] = max_price
                direction = -1
                # 新的下跌段从当前谷起步
                min_price, min_i = lo, i

        elif direction == -1:
            # 下跌段持续刷新谷值
            if lo < min_price:
                min_price, min_i = lo, i
            # 若从谷值反弹达到阈值 → 确认该谷为 swing low，方向转多
            if risen_from_min(hi, min_price) and min_i < i:
                swing_low[min_i] = min_price
                direction = 1
                # 新的上涨段从当前峰起步
                max_price, max_i = hi, i

    # 序列结束：是否保留最后候选极值
    if include_last:
        if direction == 1 and max_i is not None:
            swing_high[max_i] = max_price
        elif direction == -1 and min_i is not None:
            swing_low[min_i] = min_price
        elif direction == 0:
            # 只有一次波动都未达阈值：保留更“极端”的那个端点
            if (max_price - min_price) >= 0:
                swing_high[max_i] = max_price
            else:
                swing_low[min_i] = min_price

    return pd.DataFrame(
        {"swing_high": swing_high, "swing_low": swing_low},
        index=idx
    )



# ========= 2) 回撤段提取（高 -> 后继低） =========

PairsT = List[Tuple[pd.Timestamp, float, pd.Timestamp, float, float]]

def extract_contractions_v2(
    df: pd.DataFrame,
    swings: pd.DataFrame,
    lookback: Optional[int] = None,   # None = 全历史
    min_drop: float = 0.03,
    max_drop: float = 0.40,
    min_bars_per_leg: int = 5,
    max_last_drop: Optional[float] = None,
) -> Dict[str, object]:
    """
    在窗口内顺序匹配 (最近 swing high -> 后继第一个 swing low)，计算 drop=(hi-low)/hi，
    并过滤过小/过大的回撤与“过近”的锯齿（min_bars_per_leg）。
    返回 dict，包括 pairs、last_drop、passed_last_tight 等。
    """
    df = ensure_datetime_index(df)
    if lookback is None:
        d = df.copy()
    else:
        lookback = int(lookback)
        if lookback <= 0:
            raise ValueError("lookback 必须为正整数或 None")
        d = df.iloc[-lookback:].copy()

    if d.empty:
        return {
            "pairs": [], "last_drop": None, "passed_last_tight": True,
            "window_start": None, "window_end": None, "used_bars": 0,
        }

    sw = swings.reindex(d.index) #reindex 方法会根据传入的新索引，重新对齐（对照）原对象的索引，生成一个新的对象。

    pairs: PairsT = []
    pending_high: Optional[Tuple[pd.Timestamp, float]] = None
    last_high_i: Optional[int] = None

    for i, t in enumerate(d.index):
        hi = sw.at[t, "swing_high"] if "swing_high" in sw.columns else np.nan
        lo = sw.at[t, "swing_low"]  if "swing_low"  in sw.columns else np.nan

        if pd.notna(hi):
            pending_high = (t, float(hi))
            last_high_i = i

        if (pending_high is not None) and pd.notna(lo):
            if (last_high_i is not None) and ((i - last_high_i) < int(min_bars_per_leg)):
                # 段内太短，疑似噪声，继续等待更合适的 low
                continue

            hi_t, hi_v = pending_high
            lo_v = float(lo)

            if hi_v <= lo_v:  # 形态异常，放弃本段
                pending_high = None
                last_high_i = None
                continue

            drop = (hi_v - lo_v) / hi_v if hi_v > 0 else np.nan
            if pd.notna(drop) and (min_drop <= drop <= max_drop):
                pairs.append((hi_t, hi_v, t, lo_v, float(drop)))

            # 完成 high->(第一个)low 的配对
            pending_high = None
            last_high_i = None

    last_drop = pairs[-1][4] if pairs else None
    if pairs and (max_last_drop is not None):
        passed_last_tight = (last_drop <= max_last_drop)
    else:
        passed_last_tight = True

    return {
        "pairs": pairs,
        "last_drop": last_drop,
        "passed_last_tight": bool(passed_last_tight),
        "window_start": d.index[0] if len(d.index) else None,
        "window_end":   d.index[-1] if len(d.index) else None,
        "used_bars":    len(d.index),
    }


# ========= 3) VCP 校验（递减 + 低点抬高，含 low_tol 三模式） =========

def _resolve_low_tolerance(
    lows: List[float],
    low_times: List[pd.Timestamp],
    mode: str,
    low_abs: float,
    low_pct: float,
    low_atr_mult: float,
    atr_series: Optional[pd.Series],
    i: int
) -> float:
    """
    计算第 i->i+1 对低点所需的“最小抬高容差”（> req 才通过）。
    - mode="abs"：常数阈值（如 0.2 元）
    - mode="pct"：按前一个低点的百分比（如 1%）
    - mode="atr"：按 ATR * 倍数（需要 atr_series）
    """
    if mode == "abs":
        return float(low_abs)
    elif mode == "pct":
        base = float(lows[i])  # 以前一个低点为基准
        return float(low_pct) * base
    elif mode == "atr":
        if atr_series is None:
            raise ValueError("low_mode='atr' 需要提供 atr_series")
        # 用“后一个低点”的 ATR 作为容差基准（也可改为两者均值）
        t = low_times[i+1]
        val = atr_series.reindex([t]).iloc[0] if t in atr_series.index else np.nan
        if pd.isna(val):
            # 回退策略：若无 ATR 数据，用 abs=0 兜底
            return 0.0
        return float(low_atr_mult) * float(val)
    else:
        raise ValueError("low_mode 必须是 'abs' | 'pct' | 'atr'")


def is_valid_vcp_sequence_v2(
    seq_or_result: Union[PairsT, Dict[str, object]],
    need_n: int = 3,
    tol_drop: float = 0.0,               # 回撤递减的最小差额容差（如 0.01 = 至少差1pct）
    enforce_higher_lows: bool = True,
    # --- 低点抬高容差模式 ---
    low_mode: str = "abs",               # "abs" | "pct" | "atr"
    low_abs: float = 0.0,                # mode=abs 时使用
    low_pct: float = 0.0,                # mode=pct 时使用（如 0.01 = 1%）
    low_atr_mult: float = 0.0,           # mode=atr 时使用（如 0.5 表示 0.5 * ATR）
    atr_series: Optional[pd.Series] = None,
    # --- 末段更紧 ---
    require_last_tight: bool = False
) -> Dict[str, object]:
    """
    校验 VCP 收缩序列结构：
      1) 最近 need_n 段回撤幅度严格递减（drops[i] > drops[i+1] + tol_drop）
      2) （可选）低点抬高：lows[i+1] - lows[i] > low_tol(i)，支持 abs / pct / atr 三模式
      3) （可选）要求末段更紧（需传入 extract_contractions_v2 的返回字典才能知道）
    """
    if isinstance(seq_or_result, dict):
        pairs: PairsT = seq_or_result.get("pairs", [])  # type: ignore
        passed_last_tight = bool(seq_or_result.get("passed_last_tight", True))
        last_drop = seq_or_result.get("last_drop", None)
    else:
        pairs = seq_or_result
        passed_last_tight = True
        last_drop = pairs[-1][4] if pairs else None

    if len(pairs) < need_n:
        return {
            "valid": False, "reason": f"not_enough_pairs(<{need_n})",
            "sub_pairs": None, "drops": None, "lows": None,
            "last_drop": last_drop, "passed_last_tight": passed_last_tight,
            "pivot_time": None, "pivot_price": None,
        }

    sub = pairs[-need_n:]
    drops = [x[4] for x in sub]
    lows  = [x[3] for x in sub]
    low_times = [x[2] for x in sub]  # 低点发生时间（用来取 ATR）

    # 1) 回撤递减 + 容差
    for i in range(len(drops) - 1):
        if not (drops[i] > drops[i+1] + tol_drop):
            return {
                "valid": False,
                "reason": f"drops_not_decreasing_at_{i}: {drops[i]:.4f} <= {drops[i+1]:.4f} + {tol_drop:.4f}",
                "sub_pairs": sub, "drops": drops, "lows": lows,
                "last_drop": last_drop, "passed_last_tight": passed_last_tight,
                "pivot_time": None, "pivot_price": None,
            }

    # 2) 低点抬高 + 模式化容差
    if enforce_higher_lows:
        for i in range(len(lows) - 1):
            req = _resolve_low_tolerance(
                lows, low_times, low_mode, low_abs, low_pct, low_atr_mult, atr_series, i
            )
            if not ((lows[i+1] - lows[i]) > req):
                return {
                    "valid": False,
                    "reason": f"lows_not_increasing_at_{i}: "
                              f"{lows[i+1]:.4f} - {lows[i]:.4f} <= req({req:.4f})",
                    "sub_pairs": sub, "drops": drops, "lows": lows,
                    "last_drop": last_drop, "passed_last_tight": passed_last_tight,
                    "pivot_time": None, "pivot_price": None,
                }

    # 3) （可选）末段更紧
    if require_last_tight and (not passed_last_tight):
        return {
            "valid": False,
            "reason": "last_leg_not_tight_enough",
            "sub_pairs": sub, "drops": drops, "lows": lows,
            "last_drop": last_drop, "passed_last_tight": passed_last_tight,
            "pivot_time": None, "pivot_price": None,
        }

    # 4) 生成 pivot 候选（最近一段的 swing high）
    pivot_time = sub[-1][0]
    pivot_price = sub[-1][1]

    return {
        "valid": True, "reason": None,
        "sub_pairs": sub, "drops": drops, "lows": lows,
        "last_drop": last_drop, "passed_last_tight": passed_last_tight,
        "pivot_time": pivot_time, "pivot_price": pivot_price,
    }


# ========= 4) 无前视封装（产出与日线对齐的 vcp_ok / pivot_price） =========

def vcp_flags_no_lookahead(
    df: pd.DataFrame,
    window:int = 5,                 # 摆动识别中心窗（会延迟 half=window//2 确认）
    lookback: int = 120,            # 每个可用当日向后看的K线数量
    # ---- 回撤段过滤 ----
    min_drop: float = 0.03,
    max_drop: float = 0.40,
    min_bars_per_leg:int = 5,
    max_last_drop: float | None = None,
    # ---- VCP 校验参数 ----
    need_n:int = 3,
    tol_drop: float = 0.0,
    enforce_higher_lows: bool = True,
    # 低点抬高容差三模式
    low_mode: str = "abs",          # "abs" | "pct" | "atr"
    low_abs: float = 0.0,           # abs: 至少高出固定价差
    low_pct: float = 0.0,           # pct: 至少高出前低点的百分比（如 0.01 = 1%）
    low_atr_mult: float = 0.0,      # atr: 至少高出 ATR * 倍数
    atr_len: int = 20,              # atr 模式用到的 ATR 长度
    require_last_tight: bool = False
) -> pd.DataFrame:
    """
    输出与 df.index 对齐的两列：
      - vcp_ok: bool（在【已确认】当日才为 True；无前视）
      - pivot_price: float（最近一段 swing high 价格；无前视）
    依赖本文件内 find_swings / extract_contractions_v2 / is_valid_vcp_sequence_v2。
    """
    df = ensure_datetime_index(df)
    half = window // 2

    # 1) 摆动点（中心法）+ 延迟 half 确认
    swings_raw = find_swings(df, window=window)
    swings = swings_raw.shift(half)

    # 2) 准备 ATR（仅当 low_mode='atr' 时需要）
    atr_series = None
    if low_mode == "atr":
        atr_series = atr_simple(df, n=atr_len)

    # 3) 逐日滚动窗口检查
    vcp_ok = pd.Series(False, index=df.index)
    pivot_price_s = pd.Series(np.nan, index=df.index)

    def check_window(end_pos:int):
        start_pos = max(0, end_pos - lookback + 1)
        dwin = df.iloc[start_pos:end_pos+1]
        swin = swings.loc[dwin.index]

        res_pairs = extract_contractions_v2(
            dwin, swin,
            lookback=None,
            min_drop=min_drop, max_drop=max_drop,
            min_bars_per_leg=min_bars_per_leg,
            max_last_drop=max_last_drop
        )
        chk = is_valid_vcp_sequence_v2(
            res_pairs,
            need_n=need_n,
            tol_drop=tol_drop,
            enforce_higher_lows=enforce_higher_lows,
            low_mode=low_mode, low_abs=low_abs, low_pct=low_pct,
            low_atr_mult=low_atr_mult, atr_series=atr_series,
            require_last_tight=require_last_tight
        )
        return chk

    for i, t in enumerate(df.index):
        if i < half:
            continue
        chk = check_window(i)
        if chk["valid"]:
            vcp_ok.iloc[i] = True
            pivot_price_s.iloc[i] = chk["pivot_price"]

    out = pd.DataFrame({"vcp_ok": vcp_ok, "pivot_price": pivot_price_s}, index=df.index)
    return out


# ========= （可选）简易示例 =========
if __name__ == "__main__":
    # 构造一个极简示例（真实使用请替换为你的行情 df）
    idx = pd.date_range("2024-01-01", periods=200, freq="B")
    rng = np.random.default_rng(0)
    close = np.cumsum(rng.normal(0, 1, size=len(idx))) + 50
    high = close + rng.normal(0.5, 0.3, size=len(idx)).clip(min=0)
    low  = close - rng.normal(0.5, 0.3, size=len(idx)).clip(min=0)
    vol  = (rng.lognormal(mean=12, sigma=0.2, size=len(idx))).astype(int)

    df_demo = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)

    # 运行无前视 VCP（ATR 自适应 low_tol 举例：至少高出 0.5 * ATR20）
    res = vcp_flags_no_lookahead(
        df_demo,
        window=5, lookback=120,
        min_drop=0.03, max_drop=0.40, min_bars_per_leg=5,
        need_n=3, tol_drop=0.0, enforce_higher_lows=True,
        low_mode="atr", low_atr_mult=0.5, atr_len=20,
        require_last_tight=False
    )
    print(res.tail())
