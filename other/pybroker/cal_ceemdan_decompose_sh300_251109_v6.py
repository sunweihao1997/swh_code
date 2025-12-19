# -*- coding: utf-8 -*-
"""
CEEMDAN 双线策略回测（多指数 × 多窗口 × 多斜率回看）+ Excel 报告（收益率降序）
- 指数：沪深300/上证50/中证1000
- 滚动CEEMDAN窗口：300, 500, 700, 900
- 双线：fast/slow；候选 drop={0..5}（0=original），自动生成 fast<slow 全部组合
- 策略：金叉且慢线斜率>0（回看 1/3/5）开多；死叉平仓；信号次日生效（避免前视）
- 图表：每个测试输出一张图（价格+买卖点 / 策略收益vs持有）
- 报告：Excel 总表+分表，按年化收益率降序；若无 Excel 引擎回退 CSV
- 含：指数数据缓存 CSV、滚动分解缓存 Parquet/CSV、进度条 tqdm
"""

import os
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import akshare as ak
from PyEMD import CEEMDAN

# === tqdm 进度条（可选） ===
try:
    from tqdm.auto import tqdm
    USE_TQDM = True
except Exception:
    tqdm = None
    USE_TQDM = False

# === Parquet（可选） ===
try:
    import pyarrow  # noqa
    PARQUET_AVAILABLE = True
except Exception:
    PARQUET_AVAILABLE = False


# =========================
# 配置
# =========================
OUTPUT_DIR = "/home/sun/paint/stock/"
CACHE_DIR  = "/home/sun/paint/cache/"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

START_DATE = "2010-01-01"
END_DATE   = "2025-11-09"       # None 表示到最新

WINDOWS    = [150,200]  # 滚动CEEMDAN窗
SLOPE_LBKS = [1, 2, 3]             # 慢线斜率回看（>0才允许开多）

TRIALS     = 100
EPSILON    = 0.2
SEED       = 42

# 指数列表： (友好名称, AkShare symbol)
INDEXES = [
    ("CSI300",  "sh000300"),   # 沪深300
#    ("SSE50",   "sh000016"),   # 上证50
#    ("CSI1000", "sh000852"),   # 中证1000
]

# 候选 drop（0=original；越大越平滑）
CANDIDATE_DROPS = [0, 1, 2, 3,]

# 报告与排序字段（annual_return 或 total_return，默认按年化）
SORT_FIELD   = "annual_return"
REPORT_XLSX  = os.path.join(OUTPUT_DIR, f"report_dualline_{START_DATE}_{END_DATE or 'latest'}_sorted_by_{SORT_FIELD}.xlsx")
REPORT_CSV   = os.path.join(OUTPUT_DIR, f"report_dualline_{START_DATE}_{END_DATE or 'latest'}_sorted_by_{SORT_FIELD}.csv")


# =========================
# 工具：缓存文件名
# =========================
def _hash_str(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]


def data_cache_path(symbol, start_date, end_date):
    tag = f"{symbol}_{start_date}_{end_date or 'latest'}"
    return os.path.join(CACHE_DIR, f"idx_{tag}.csv")


def trend_cache_paths(symbol, start_date, end_date, window, trials, epsilon, seed, drops, n_points):
    base = f"{symbol}_W{window}_T{trials}_E{epsilon}_S{seed}_{start_date}_{end_date or 'latest'}_D{','.join(map(str,sorted(drops)))}_N{n_points}"
    tag = _hash_str(base)
    pq  = os.path.join(CACHE_DIR, f"roll_{tag}.parquet")
    csv = os.path.join(CACHE_DIR, f"roll_{tag}.csv")
    meta= os.path.join(CACHE_DIR, f"roll_{tag}.meta.json")
    return pq, csv, meta


# =========================
# 数据获取（带缓存）
# =========================
def load_index_cached(symbol, start_date, end_date, cache_csv, force_refresh=False):
    if (not force_refresh) and os.path.exists(cache_csv):
        df = pd.read_csv(cache_csv, parse_dates=["date"])
        df = df[(df["date"] >= pd.to_datetime(start_date))]
        if end_date is not None:
            df = df[(df["date"] <= pd.to_datetime(end_date))]
        if not df.empty:
            close = df["close"].astype(float).reset_index(drop=True)
            dates = df["date"].reset_index(drop=True)
            print(f"[DataCache] {symbol}: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()} ({len(close)} pts) loaded from cache")
            return close, dates

    df = ak.stock_zh_index_daily(symbol=symbol)
    if df is None or df.empty:
        raise RuntimeError(f"Failed to fetch index data: {symbol}")

    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        date_col = "date"
    else:
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={"index": "date"})
        date_col = "date"

    if "close" not in df.columns:
        raise RuntimeError(f"'close' column not in data for {symbol}. got: {df.columns}")

    df = df[(df[date_col] >= pd.to_datetime(start_date))]
    if end_date is not None:
        df = df[(df[date_col] <= pd.to_datetime(end_date))]

    df = df[[date_col, "close"]].reset_index(drop=True)
    df.to_csv(cache_csv, index=False)

    close = df["close"].astype(float).copy()
    dates = df[date_col].copy()
    print(f"[DataCache] {symbol}: downloaded & cached | {dates.iloc[0].date()} ~ {dates.iloc[-1].date()} ({len(close)} pts)")
    return close, dates


# =========================
# CEEMDAN 基础
# =========================
def ceemdan_decompose(values, trials=TRIALS, epsilon=EPSILON, seed=SEED):
    values = np.asarray(values, dtype=float)
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)
    ceemdan.noise_seed(seed)
    imfs = ceemdan.ceemdan(values)
    residue = None
    try:
        imfs2, residue2 = ceemdan.get_imfs_and_residue()
        if np.asarray(imfs2).shape == imfs.shape:
            residue = np.asarray(residue2, dtype=float)
    except AttributeError:
        pass
    if residue is None:
        residue = values - imfs.sum(axis=0)
    return imfs, residue


def reconstruct_last_point(imfs, residue, drop_first_n):
    n_imf = imfs.shape[0]
    d = int(max(0, min(drop_first_n, n_imf)))
    if d == n_imf:
        return float(residue[-1])
    return float(np.sum(imfs[d:, -1]) + residue[-1])


# =========================
# 滚动 CEEMDAN（可缓存）
# =========================
def save_trend_cache(df_trend, pq, csv, meta_path, meta):
    try:
        if PARQUET_AVAILABLE:
            df_trend.to_parquet(pq, index=False)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[TrendCache] saved -> {pq}")
        else:
            df_trend.to_csv(csv, index=False)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[TrendCache] saved -> {csv}")
    except Exception as e:
        df_trend.to_csv(csv, index=False)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"[TrendCache] saved -> {csv} (fallback). reason={e}")


def load_trend_cache(pq, csv, meta_path):
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[TrendCache] loaded <- {pq}")
        return df, meta
    if os.path.exists(csv):
        df = pd.read_csv(csv, parse_dates=["date"])
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        print(f"[TrendCache] loaded <- {csv}")
        return df, meta
    return None, None


def compute_rolling_trends_lastpoint(prices, dates, window, drop_values,
                                     trials=TRIALS, epsilon=EPSILON, seed=SEED,
                                     show_progress=True):
    values = np.asarray(prices, dtype=float)
    N = len(values)
    drops = sorted(set(int(x) for x in drop_values if x != 0))  # 0=original
    trend_map = {d: np.full(N, np.nan, dtype=float) for d in drops}
    trend_map0 = np.asarray(values, dtype=float)  # drop_0 = original

    rng = range(window - 1, N)
    iterator = rng
    total = N - (window - 1)
    if show_progress and USE_TQDM and tqdm is not None:
        iterator = tqdm(rng, total=total, desc=f"Rolling CEEMDAN (W={window})",
                        unit="win", dynamic_ncols=True, mininterval=0.5)

    for idx, t in enumerate(iterator):
        win = values[t - window + 1: t + 1]
        imfs, residue = ceemdan_decompose(win, trials=trials, epsilon=epsilon, seed=seed)
        for d in drops:
            trend_map[d][t] = reconstruct_last_point(imfs, residue, d)

        if USE_TQDM and tqdm is not None and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(end=str(pd.to_datetime(dates.iloc[t]).date()),
                                 imfs=imfs.shape[0], trials=trials, eps=epsilon)
        elif (not USE_TQDM or tqdm is None) and (idx % 100 == 0):
            done = idx + 1
            print(f"[{done:>5}/{total}] end={dates.iloc[t].date()}, IMFs={imfs.shape[0]}")

    df_trend = pd.DataFrame({"date": pd.to_datetime(dates)})
    df_trend["drop_0"] = trend_map0
    for d in drops:
        df_trend[f"drop_{d}"] = trend_map[d]
    return df_trend


def load_or_compute_trends(symbol, values, dates, start_date, end_date, window, drops,
                           trials, epsilon, seed, force_recompute=False):
    pq, csv, meta_path = trend_cache_paths(symbol, start_date, end_date, window, trials, epsilon, seed, drops, len(values))

    if not force_recompute:
        df_cached, meta = load_trend_cache(pq, csv, meta_path)
        if df_cached is not None:
            if len(df_cached) == len(dates) and np.all(pd.to_datetime(df_cached["date"]).values == pd.to_datetime(dates).values):
                need_cols = [f"drop_{d}" for d in sorted(set(drops))]
                if all(c in df_cached.columns for c in need_cols):
                    print(f"[TrendCache] reuse cached trends for {symbol} (W={window})")
                    return df_cached

    print(f"[TrendCache] compute trends for {symbol} (W={window})")
    df_trend = compute_rolling_trends_lastpoint(values, dates, window, drops,
                                                trials=trials, epsilon=epsilon, seed=seed,
                                                show_progress=True)
    meta = dict(symbol=symbol, start_date=start_date, end_date=end_date,
                window=window, drops=sorted(set(drops)),
                trials=trials, epsilon=epsilon, seed=seed, n=len(values))
    save_trend_cache(df_trend, pq, csv, meta_path, meta)
    return df_trend


# =========================
# 策略（双线，金叉+慢线斜率>0；死叉平仓；次日生效）
# =========================
def run_dual_line_strategy(prices, fast_line, slow_line, slope_lookback=1):
    P = np.asarray(prices, dtype=float)
    F = np.asarray(fast_line, dtype=float)
    S = np.asarray(slow_line, dtype=float)
    N = len(P)

    position = np.zeros(N, dtype=int)
    buy_idx, sell_idx = [], []

    start_t = max(1, slope_lookback)

    for t in range(start_t, N - 1):
        position[t + 1] = position[t]

        if not (np.isfinite(F[t]) and np.isfinite(S[t]) and
                np.isfinite(F[t-1]) and np.isfinite(S[t-1]) and
                np.isfinite(S[t - slope_lookback])):
            continue

        cross_up = (F[t-1] <= S[t-1]) and (F[t] > S[t])
        cross_dn = (F[t-1] >= S[t-1]) and (F[t] < S[t])
        slow_up  = (S[t] > S[t - slope_lookback])  # 慢线斜率>0

        # 死叉优先：次日平仓
        if cross_dn and position[t] == 1:
            position[t + 1] = 0
            sell_idx.append(t + 1)
            continue

        # 金叉且慢线向上：次日开多
        if cross_up and slow_up and position[t] == 0:
            position[t + 1] = 1
            buy_idx.append(t + 1)

    # 收益（对数，前一日仓位作用于次日）
    log_ret = np.zeros(N)
    log_ret[1:] = np.log(P[1:] / P[:-1])
    strat_ret = np.zeros(N)
    strat_ret[1:] = position[:-1] * log_ret[1:]

    equity = np.ones(N)
    equity[1:] = np.exp(np.cumsum(strat_ret[1:]))

    bh_equity = np.ones(N)
    bh_equity[1:] = np.exp(np.cumsum(log_ret[1:]))

    return position, np.array(buy_idx, dtype=int), np.array(sell_idx, dtype=int), equity, bh_equity


# =========================
# 绩效
# =========================
def perf_report(equity, trading_days_per_year=252):
    equity = np.asarray(equity, dtype=float)
    total_return = equity[-1] / equity[0] - 1.0
    ret = equity[1:] / equity[:-1] - 1.0
    years = len(ret) / trading_days_per_year
    annual = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    return total_return, annual


# =========================
# 绘图（每个测试一张图：上=价格+买卖点，下=收益曲线）
# =========================
def plot_single_figure(prices, dates, buy_idx, sell_idx, equity, bh_equity, title, save_path):
    dates = pd.to_datetime(dates)
    P = np.asarray(prices, dtype=float)

    fig = plt.figure(figsize=(13, 7))

    # Top: price + signals
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(dates, P, label="Index", linewidth=1.0)
    if len(buy_idx) > 0:
        ax1.scatter(dates[buy_idx], P[buy_idx], marker="^", s=40, label="Buy", alpha=0.85)
    if len(sell_idx) > 0:
        ax1.scatter(dates[sell_idx], P[sell_idx], marker="v", s=40, label="Sell", alpha=0.85)
    ax1.set_ylabel("Index")
    ax1.set_title(title)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left")

    # Bottom: equity
    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(dates, bh_equity, label="Buy & Hold", linewidth=1.0)
    ax2.plot(dates, equity, label="Strategy", linewidth=1.2)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Equity (normalized)")
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left")

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"[Plot] {save_path}")


# =========================
# 主流程
# =========================
def main(force_refresh_data=False, force_recompute_trends=False):
    results = []  # 汇总到 Excel

    # 生成所有 fast<slow 组合
    drops = sorted(set(CANDIDATE_DROPS))
    pairs = [(f, s) for i, f in enumerate(drops) for s in drops[i+1:]]

    for name, symbol in INDEXES:
        print(f"\n=== {name} ({symbol}) ===")
        cache_csv = data_cache_path(symbol, START_DATE, END_DATE)
        close, dates = load_index_cached(symbol, START_DATE, END_DATE, cache_csv, force_refresh=force_refresh_data)
        values = close.values

        # 外层迭代窗口
        win_iter = WINDOWS
        if USE_TQDM and tqdm is not None:
            win_iter = tqdm(WINDOWS, desc=f"{name} windows", dynamic_ncols=True)

        for W in win_iter:
            out_dir_idx = os.path.join(OUTPUT_DIR, f"{name}_W{W}")
            os.makedirs(out_dir_idx, exist_ok=True)

            # 载入/计算该窗口的滚动趋势（包含 drop_0）
            df_trend = load_or_compute_trends(symbol, values, dates, START_DATE, END_DATE,
                                              W, drops, TRIALS, EPSILON, SEED,
                                              force_recompute=force_recompute_trends)

            # 组合回测 × 斜率回看
            pair_iter = pairs
            if USE_TQDM and tqdm is not None:
                pair_iter = tqdm(pairs, desc=f"{name} W{W} combos", leave=False, dynamic_ncols=True)

            for fast_d, slow_d in pair_iter:
                fast = df_trend[f"drop_{fast_d}"].to_numpy()
                slow = df_trend[f"drop_{slow_d}"].to_numpy()

                for lbk in SLOPE_LBKS:
                    position, buy_idx, sell_idx, equity, bh_equity = run_dual_line_strategy(
                        values, fast, slow, slope_lookback=lbk
                    )
                    total, annual = perf_report(equity)

                    # 记录
                    results.append({
                        "index": name,
                        "symbol": symbol,
                        "window": W,
                        "fast_drop": fast_d,
                        "slow_drop": slow_d,
                        "slope_lookback": lbk,
                        "trades": int(len(buy_idx)),
                        "total_return": float(total),
                        "annual_return": float(annual),
                    })

                    tag = f"{name}_W{W}_d{fast_d}-{slow_d}_lbk{lbk}"
                    fig_path = os.path.join(out_dir_idx, f"test_{tag}.png")
                    plot_single_figure(
                        prices=values,
                        dates=dates,
                        buy_idx=buy_idx,
                        sell_idx=sell_idx,
                        equity=equity,
                        bh_equity=bh_equity,
                        title=f"{name} (W={W}, fast=drop{fast_d}, slow=drop{slow_d}, lbk={lbk})",
                        save_path=fig_path
                    )

    # === 生成报告（按收益率降序）===
    df_report = pd.DataFrame(results)
    if df_report.empty:
        print("[Report] No results. Check data or parameters.")
        return
    df_report.sort_values(by=[SORT_FIELD], ascending=False, inplace=True)

    # Excel 优先，失败回退 CSV
    try:
        engine = None
        try:
            import xlsxwriter  # noqa
            engine = "xlsxwriter"
        except Exception:
            engine = "openpyxl"

        with pd.ExcelWriter(REPORT_XLSX, engine=engine) as writer:
            df_report.to_excel(writer, sheet_name="Summary", index=False)
            # 各指数分表
            for name, _ in INDEXES:
                sub = df_report[df_report["index"] == name]
                if not sub.empty:
                    sub.to_excel(writer, sheet_name=name, index=False)
        print(f"[Report] Excel written -> {REPORT_XLSX}")
    except Exception as e:
        df_report.to_csv(REPORT_CSV, index=False)
        print(f"[Report] Excel failed ({e}). Fallback CSV -> {REPORT_CSV}")

    print("\nAll done.")


if __name__ == "__main__":
    # force_refresh_data=True      强制重新下载指数数据（忽略CSV缓存）
    # force_recompute_trends=True  强制重新计算滚动CEEMDAN（忽略Parquet/CSV缓存）
    main(force_refresh_data=False, force_recompute_trends=False)
