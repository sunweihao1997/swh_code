"""
2025-09-07 最终合并版（候选 AND/OR 切换 + 窗口限定 + 重叠缓冲 + 可视化父窗口 & 命中窗口）
- 修复 DatetimeIndex & 列名统一
- ATR 收缩（周-日联动）→ 生成执行窗口（exec_window_days）
- ZigZag 拐点 → 回撤段 → VCP 校验
- 仅在“执行窗口内”（含缓冲 & 重叠阈值）扫描 VCP 命中
- 候选阶段支持 AND/OR 两种模式（candidate_mode='and'|'or'）
- 可视化：价格 + ZigZag + 全部 pairs + 父窗口(浅灰) + 命中窗口(绿色) + pivot 虚线
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cal_stock_ATR_shrinking_250901 import (
    atr_ta, bb_width_ta, low_volatility_flags, link_week_daily,
    to_weekly, percentile_rank_last
)
from cal_stock_VCP_Algorithm_v2_250904 import (
    find_swings_zigzag, extract_contractions_v2, is_valid_vcp_sequence_v2
)

# ======================== 基础工具 ========================
def force_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """强制转为 DatetimeIndex，并删除遗留的 'date' 列。"""
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
    if "date" in _df.columns:
        _df = _df.drop(columns=["date"])
    return _df

# ======================== 候选阶段：AND / OR 两种模式 ========================
def low_volatility_flags_or(df_daily: pd.DataFrame,
                            d_len: int = 20, w_len: int = 20,
                            d_lookback_days: int = 252,   # 日线分位窗口
                            w_lookback_weeks: int = 52,    # 周线分位窗口
                            d_pct_atr: float = 0.35, d_pct_bbw: float = 0.35,
                            w_pct_atr: float = 0.50, w_pct_bbw: float = 0.50,
                            bb_std: float = 2.0) -> dict:
    """OR 逻辑：ATR 或 BBW 任一低于阈值即可。"""
    df_daily = force_datetime_index(df_daily).copy()

    # 日线
    d = df_daily.copy()
    d[f'ATR{d_len}_d'] = atr_ta(d, d_len)
    d[f'BBW{d_len}_d'] = bb_width_ta(d, d_len, std=bb_std)
    d['rank_ATR_d'] = percentile_rank_last(d[f'ATR{d_len}_d'], d_lookback_days)
    d['rank_BBW_d'] = percentile_rank_last(d[f'BBW{d_len}_d'], d_lookback_days)
    d['daily_low_vol'] = (d['rank_ATR_d'] <= d_pct_atr) | (d['rank_BBW_d'] <= d_pct_bbw)

    # 周线
    w = to_weekly(df_daily)
    w[f'ATR{w_len}_w'] = atr_ta(w, w_len)
    w[f'BBW{w_len}_w'] = bb_width_ta(w, w_len, std=bb_std)
    w['rank_ATR_w'] = percentile_rank_last(w[f'ATR{w_len}_w'], w_lookback_weeks)
    w['rank_BBW_w'] = percentile_rank_last(w[f'BBW{w_len}_w'], w_lookback_weeks)
    w['weekly_low_vol'] = (w['rank_ATR_w'] <= w_pct_atr) | (w['rank_BBW_w'] <= w_pct_bbw)

    return {'daily': d, 'weekly': w}

def build_candidates(df_daily: pd.DataFrame,
                     mode: str = "and",
                     *,
                     d_len: int = 20, w_len: int = 20,
                     d_lookback_days: int = 252, w_lookback_weeks: int = 52,
                     d_pct: float = 0.20, w_pct: float = 0.35,
                     d_pct_atr: float = 0.35, d_pct_bbw: float = 0.35,
                     w_pct_atr: float = 0.50, w_pct_bbw: float = 0.50,
                     bb_std: float = 2.0,
                     exec_window_days: int = 20,
                     start_next_day: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    mode='and'  → 使用原 AND 逻辑：min(rank(ATR), rank(BBW)) ≤ 阈值
    mode='or'   → 使用 OR 逻辑：rank(ATR) ≤ 阈值 或 rank(BBW) ≤ 阈值
    返回：(daily_flags, weekly_flags, candidates_series)
    """
    df_daily = force_datetime_index(df_daily)
    if mode.lower() == "and":
        flags = low_volatility_flags(
            df_daily=df_daily,
            d_len=d_len, w_len=w_len,
            d_lookback_days=d_lookback_days, w_lookback_weeks=w_lookback_weeks,
            d_pct=d_pct, w_pct=w_pct,
            bb_std=bb_std
        )
    elif mode.lower() == "or":
        flags = low_volatility_flags_or(
            df_daily=df_daily,
            d_len=d_len, w_len=w_len,
            d_lookback_days=d_lookback_days, w_lookback_weeks=w_lookback_weeks,
            d_pct_atr=d_pct_atr, d_pct_bbw=d_pct_bbw,
            w_pct_atr=w_pct_atr, w_pct_bbw=w_pct_bbw,
            bb_std=bb_std
        )
    else:
        raise ValueError("candidate_mode 必须是 'and' 或 'or'。")

    daily_flags = flags['daily']
    weekly_flags = flags['weekly']
    candidates = link_week_daily(
        daily_df=df_daily,
        flags_d=daily_flags,
        flags_w=weekly_flags,
        exec_window_days=exec_window_days,
        start_next_day=start_next_day
    )
    return daily_flags, weekly_flags, candidates

# ======================== 可视化（候选日） ========================
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

# ======================== 执行窗口工具 ========================
def windows_from_candidates(candidates: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """从 candidates（布尔Series）提取连续 True 的时间段。"""
    c = candidates.copy().fillna(False)
    runs = []
    in_run = False
    start = None
    prev_t = None
    for t, flag in c.items():
        if flag and not in_run:
            in_run = True
            start = t
        elif not flag and in_run:
            runs.append((start, prev_t))
            in_run = False
        prev_t = t
    if in_run and prev_t is not None:
        runs.append((start, prev_t))
    return runs  # [(start_ts, end_ts), ...]

def pairs_overlap_window(
    pairs: list,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    pre_buffer_days: int = 5,
    post_buffer_days: int = 5,
    min_overlap_ratio: float = 0.5
) -> list:
    """
    选出与窗口 [start, end] 发生“足够重叠”的 high→low 段。
    - 窗口左右各加 buffer（默认前后各 5 天）
    - 重叠比例 = overlap / 段时长（相对 hi_t→lo_t 的天数），默认至少 50%
    """
    ws = start - pd.Timedelta(days=pre_buffer_days)
    we = end + pd.Timedelta(days=post_buffer_days)

    kept = []
    for (hi_t, hi_v, lo_t, lo_v, drop) in pairs:
        seg_start, seg_end = hi_t, lo_t
        if seg_end < ws or seg_start > we:
            continue
        ov_start = max(seg_start, ws)
        ov_end   = min(seg_end, we)
        overlap = (ov_end - ov_start).days
        seg_len = max((seg_end - seg_start).days, 1)
        if overlap / seg_len >= min_overlap_ratio:
            kept.append((hi_t, hi_v, lo_t, lo_v, drop))
    return kept

def find_vcp_in_windows(
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
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
    pre_buffer_days: int = 5,
    post_buffer_days: int = 5,
    min_overlap_ratio: float = 0.5,
) -> list[dict]:
    """仅在给定执行窗口（含缓冲）里扫描 VCP 命中。"""
    hits = []
    for (ws, we) in windows:
        pw = pairs_overlap_window(
            pairs, ws, we,
            pre_buffer_days=pre_buffer_days,
            post_buffer_days=post_buffer_days,
            min_overlap_ratio=min_overlap_ratio
        )
        if len(pw) < need_n:
            continue
        for i in range(len(pw) - need_n + 1):
            sub = pw[i:i+need_n]
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
                hits.append({
                    "start": sub[0][0],
                    "end":   sub[-1][2],
                    "pivot_time": chk["pivot_time"],
                    "pivot_price": chk["pivot_price"],
                    "drops": chk["drops"],
                    "lows":  chk["lows"],
                    "sub_pairs": sub,
                    "parent_window": (ws, we),
                })
    return hits

# ======================== 全历史（5年）扫描工具（对照） ========================
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
                "start": sub[0][0],
                "end":   sub[-1][2],
                "pivot_time": chk["pivot_time"],
                "pivot_price": chk["pivot_price"],
                "drops": chk["drops"],
                "lows":  chk["lows"],
                "sub_pairs": sub,
            })
    return out

# ======================== 可视化：pairs + 父窗口 + 命中窗口 ========================
def plot_vcp_signals_with_pairs_and_windows(
    df: pd.DataFrame,
    zigzag_swings: pd.DataFrame,
    res_pairs: dict,
    parent_windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    hit_windows: list[dict],
    stock_code: str,
    title: str = "VCP — Pairs & Valid Windows (Exec Window)",
    outpath: str | None = None,
    max_pairs_to_draw: int = 120,
):
    d = df.copy()
    close = d["close"]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(close.index, close.values, label="Close", linewidth=1.2)

    # 父窗口（执行窗口期）浅灰区
    for (ws, we) in parent_windows:
        ax.axvspan(ws, we, color="grey", alpha=0.08, zorder=0)

    # ZigZag 拐点
    zz = zigzag_swings.dropna(how="all")
    highs = zz["swing_high"].dropna()
    lows  = zz["swing_low"].dropna()
    if not highs.empty:
        ax.scatter(highs.index, close.loc[highs.index], s=22, marker="^", label="ZZ High", alpha=0.6)
    if not lows.empty:
        ax.scatter(lows.index,  close.loc[lows.index],  s=22, marker="v", label="ZZ Low",  alpha=0.6)

    # 全部 high→low 段
    pairs = res_pairs.get("pairs", [])
    if pairs:
        draw_pairs = pairs[-max_pairs_to_draw:]
        for (hi_t, hi_v, lo_t, lo_v, drop) in draw_pairs:
            ax.plot([hi_t, lo_t], [hi_v, lo_v], linewidth=1.0, alpha=0.65)

    # 命中窗口（窗口内满足 VCP 的子窗口）绿色半透明 + pivot 短虚线
    for w in hit_windows:
        ax.axvspan(w["start"], w["end"], color="tab:green", alpha=0.15)
        pt, pp = w["pivot_time"], w["pivot_price"]
        if pd.notna(pp):
            if isinstance(pt, pd.Timestamp):
                ax.hlines(pp, xmin=pt, xmax=pt + pd.Timedelta(days=1),
                          linestyles="dotted", linewidth=1.1, label=None)

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

    # 候选模式与阈值（你可在此处切换 AND / OR 并调参）
    candidate_mode = "or"  # "and" 或 "or"

    # AND 模式阈值（min(rank_ATR, rank_BBW) ≤ 阈值）
    and_d_pct = 0.20
    and_w_pct = 0.35

    # OR 模式阈值（ATR≤阈值 或 BBW≤阈值）
    or_d_pct_atr, or_d_pct_bbw = 0.30, 0.30
    or_w_pct_atr, or_w_pct_bbw = 0.40, 0.40

    # 执行窗口与扫描容差
    exec_window_days = 20         # 周线触发后，日线执行窗口
    pre_buf, post_buf = 15, 15      # 窗口左右缓冲天数
    min_overlap_ratio = 0.5       # 段与（扩展后）窗口的最小重叠比例

    # ZigZag/配对过滤
    zigzag_threshold = 0.03       # 0.03~0.05 可调
    min_drop, max_drop = 0.05, 0.45
    min_bars_per_leg = 1
    require_last_tight = False
    max_last_drop = None          # 例如 0.10（末段更紧）

    # VCP 严格度
    need_n = 3
    tol_drop = 0.0
    low_mode = "atr"
    low_atr_mult = 0.5            # 低点抬高容差 = 0.5 * ATR20

    # ---------- 读取 & 统一索引 ----------
    raw = pd.read_excel(stock_path, parse_dates=True)
    stock_data = force_datetime_index(raw)
    stock_data.columns = [c.lower() for c in stock_data.columns]
    for col in ["high", "low", "close"]:
        if col not in stock_data.columns:
            raise ValueError(f"缺少必需列: {col}")

    # ---------- 候选生成（AND / OR 切换） ----------
    if candidate_mode.lower() == "and":
        daily_flags, weekly_flags, candidates = build_candidates(
            stock_data, mode="and",
            d_pct=and_d_pct, w_pct=and_w_pct,
            exec_window_days=exec_window_days, start_next_day=True
        )
    else:
        daily_flags, weekly_flags, candidates = build_candidates(
            stock_data, mode="or",
            d_pct_atr=or_d_pct_atr, d_pct_bbw=or_d_pct_bbw,
            w_pct_atr=or_w_pct_atr, w_pct_bbw=or_w_pct_bbw,
            exec_window_days=exec_window_days, start_next_day=True
        )
    stock_data = stock_data.join(candidates.rename("candidate"))

    # 父窗口（候选 True 的连续区间）
    cand_windows = windows_from_candidates(candidates)
    lens = [(we - ws).days + 1 for (ws, we) in cand_windows]
    print(f"[exec windows] count={len(cand_windows)}, median_len={np.median(lens) if lens else 0}, max_len={max(lens) if lens else 0}")

    # ---------- ZigZag 拐点 ----------
    zigzag_result = find_swings_zigzag(
        df=stock_data,
        threshold=zigzag_threshold,
        mode="percent",
        use_high_low=True,
        include_last=True
    ).reindex(stock_data.index)

    # ---------- 提取回撤段 ----------
    res_pairs = extract_contractions_v2(
        df=stock_data,
        swings=zigzag_result,
        lookback=None,
        min_drop=min_drop,
        max_drop=max_drop,
        min_bars_per_leg=min_bars_per_leg,
        max_last_drop=max_last_drop
    )
    print(f"[VCP] pairs={len(res_pairs['pairs'])}, last_drop={res_pairs['last_drop']}")

    # ---------- 诊断最近窗口 ----------
    atr20 = atr_ta(stock_data, n=20)
    recent_chk = is_valid_vcp_sequence_v2(
        res_pairs, need_n=need_n, tol_drop=tol_drop,
        enforce_higher_lows=True,
        low_mode=low_mode, low_atr_mult=low_atr_mult, atr_series=atr20,
        require_last_tight=require_last_tight
    )
    print(f"[VCP] recent valid={recent_chk['valid']} reason={recent_chk['reason']}")

    # ---------- 全历史（过去5年）命中（对照） ----------
    end_date = stock_data.index.max()
    start_5y = end_date - pd.DateOffset(years=5)
    all_windows = find_all_vcp_windows(
        res_pairs["pairs"], atr_series=atr20,
        need_n=need_n, tol_drop=tol_drop,
        enforce_higher_lows=True,
        low_mode=low_mode, low_abs=0.0, low_pct=0.0, low_atr_mult=low_atr_mult,
        require_last_tight=require_last_tight
    )
    all_windows_5y = [w for w in all_windows if w["end"] >= start_5y]
    print(f"[VCP] valid windows in last 5 years (ALL): {len(all_windows_5y)}")
    for w in all_windows_5y:
        print(f"  - {w['start'].date()} → {w['end'].date()}  pivot={w['pivot_price']:.2f}")

    # ---------- 仅在执行窗口里扫描命中（重叠+缓冲） ----------
    hits_in_exec = find_vcp_in_windows(
        windows=cand_windows,
        pairs=res_pairs["pairs"],
        atr_series=atr20,
        need_n=need_n, tol_drop=tol_drop,
        enforce_higher_lows=True,
        low_mode=low_mode, low_abs=0.0, low_pct=0.0, low_atr_mult=low_atr_mult,
        require_last_tight=require_last_tight,
        pre_buffer_days=pre_buf, post_buffer_days=post_buf,
        min_overlap_ratio=min_overlap_ratio
    )
    hits_in_exec_5y = [h for h in hits_in_exec if h["end"] >= start_5y]
    print(f"[VCP within exec_window] valid windows in last 5 years: {len(hits_in_exec_5y)}")
    for h in hits_in_exec_5y:
        print(f"  - {h['start'].date()} → {h['end'].date()}  pivot={h['pivot_price']:.2f}  "
              f"(parent window: {h['parent_window'][0].date()}→{h['parent_window'][1].date()})")

    # ---------- 可视化 ----------
    # 图：候选散点
    plot_candidates(stock_data, candidates, stock_code, title=f"{stock_code} — ATR Shrink Candidates ({candidate_mode.upper()})")

    # 图：只高亮“执行窗口内命中”的 VCP（浅灰父窗口 + 绿色命中）
    out_img = f"/home/sun/paint/stock/{stock_code}_vcp_pairs_windows_{candidate_mode}_exec{exec_window_days}_5y.png"
    plot_vcp_signals_with_pairs_and_windows(
        df=stock_data,
        zigzag_swings=zigzag_result,
        res_pairs=res_pairs,
        parent_windows=cand_windows,
        hit_windows=hits_in_exec_5y,
        stock_code=stock_code,
        title=f"{stock_code} — VCP in exec_window ({candidate_mode.upper()}, Last 5y)",
        outpath=out_img,
        max_pairs_to_draw=120
    )

    # ---------- 可选导出 ----------
    if hits_in_exec_5y:
        out_csv = f"/home/sun/paint/stock/{stock_code}_vcp_hits_{candidate_mode}_exec{exec_window_days}_5y.csv"
        pd.DataFrame([{
            "start": h["start"].date(),
            "end": h["end"].date(),
            "pivot_time": h["pivot_time"].date() if isinstance(h["pivot_time"], pd.Timestamp) else "",
            "pivot_price": h["pivot_price"],
            "drops": "|".join(f"{x:.4f}" for x in h["drops"]),
            "lows":  "|".join(f"{x:.3f}" for x in h["lows"]),
            "parent_start": h["parent_window"][0].date(),
            "parent_end": h["parent_window"][1].date(),
        } for h in hits_in_exec_5y]).to_csv(out_csv, index=False)
        print(f"Exported: {out_csv}")

    # ---------- 摘要 ----------
    print("\n===== SUMMARY =====")
    print(f"Candidate mode: {candidate_mode.upper()}  exec_window_days={exec_window_days}  buffer=({pre_buf},{post_buf})  overlap≥{min_overlap_ratio}")
    print("INDEX:", stock_data.index.dtype, stock_data.index.is_monotonic_increasing, stock_data.index.has_duplicates)
    print("swings aligned?", zigzag_result.index.equals(stock_data.index))
    print("swing_high:", zigzag_result["swing_high"].notna().sum(),
          "swing_low:",  zigzag_result["swing_low"].notna().sum())
    sh = zigzag_result["swing_high"].dropna()
    if not sh.empty:
        print("range:", sh.index.min().date(), "→", sh.index.max().date())
    print(f"Saved figure: {out_img}")
