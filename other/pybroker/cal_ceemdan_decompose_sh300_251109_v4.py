import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import akshare as ak
from PyEMD import CEEMDAN


# =========================
# 0. Config
# =========================
OUTPUT_DIR = "/home/sun/paint/stock/"
START_DATE = "2010-01-01"
END_DATE = "2025-11-09"  # 可改为 None 表示到最新
WINDOW = 1000            # 滚动 CEEMDAN 窗口长度（交易日）
K_LONG = 10              # 判定长期线向上使用的滞后天数
TRIALS = 100             # CEEMDAN 参数
EPSILON = 0.2
SEED = 42

# 要测试的 (fast_drop, mid_drop, slow_drop) 组合；需满足 fast < mid < slow
DROP_TRIPLETS = [
    (3, 4, 5),
    (3, 4, 6),
    (2, 4, 5),
]


# =========================
# 1. Data loader
# =========================
def get_hs300_close(start_date, end_date=None):
    df = ak.stock_zh_index_daily(symbol="sh000300")
    if df is None or df.empty:
        raise RuntimeError("Failed to fetch CSI 300 data from Sina via AkShare.")
    # 统一日期列
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
        raise RuntimeError(f"'close' column not found. Actual columns: {df.columns}")

    df = df[(df[date_col] >= pd.to_datetime(start_date))]
    if end_date is not None:
        df = df[(df[date_col] <= pd.to_datetime(end_date))]
    if df.empty:
        raise RuntimeError("No data after date filtering.")

    df = df.reset_index(drop=True)
    dates = df[date_col].copy()
    close = df["close"].astype(float).copy()
    print(f"Data range: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()} ({len(close)} pts)")
    return close, dates


# =========================
# 2. CEEMDAN & Reconstruction
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
    # 重构校验（可注释以提速）
    # recon = imfs.sum(axis=0) + residue
    # err = np.max(np.abs(values - recon))
    # print(f"IMFs: {imfs.shape[0]} | Recon err: {err:.2e}")
    return imfs, residue


def reconstruct_last_point(imfs, residue, drop_first_n):
    """
    仅返回重构后的最后一个点（窗口右端），以节省内存和计算
    """
    n_imf = imfs.shape[0]
    d = int(max(0, min(drop_first_n, n_imf)))
    if d == n_imf:
        return float(residue[-1])
    # 只对最后一个样本求和
    return float(np.sum(imfs[d:, -1]) + residue[-1])


# =========================
# 3. Rolling CEEMDAN (一次分解，支持多 drop)
# =========================
def compute_rolling_trends_lastpoint(prices, window, drop_values,
                                     trials=TRIALS, epsilon=EPSILON, seed=SEED):
    """
    对每个 t（从 window-1 到 N-1），用过去 window 天数据 CEEMDAN；
    对于给定的 drop_values 列表，计算每个 drop 在当日的“重构趋势值”（仅最后一点）。

    返回: dict {drop: ndarray(N, dtype=float)}，前 window-1 位置为 NaN
    """
    values = np.asarray(prices, dtype=float)
    N = len(values)
    drop_values = sorted(set(int(x) for x in drop_values))
    trend_map = {d: np.full(N, np.nan, dtype=float) for d in drop_values}

    for t in range(window - 1, N):
        win = values[t - window + 1: t + 1]
        imfs, residue = ceemdan_decompose(win, trials=trials, epsilon=epsilon, seed=seed)
        for d in drop_values:
            trend_map[d][t] = reconstruct_last_point(imfs, residue, d)

        # 简单的进度提示
        if (t - (window - 1)) % 100 == 0:
            done = t - (window - 1)
            print(f"Rolling CEEMDAN: {done}/{N - (window - 1)} windows processed...")

    return trend_map


# =========================
# 4. Strategy (Long-only with flat)
# =========================
def run_strategy_long_flat(prices, dates, T1, T2, T3, k_long=K_LONG):
    """
    根据三条趋势线（按日对齐，可能含 NaN）生成仓位与交易点。
    规则：
    - RegimeLong[t] = (T3[t] > T3[t-k_long]) & (T2[t] > T3[t]) & (T1[t] > T2[t])
    - 进场（t -> t+1执行）：RegimeLong[t] 为真 且 P[t-1] <= T1[t-1] 且 P[t] > T1[t]
    - 出场（t -> t+1执行）：RegimeLong[t] 为假 或 P[t] < T1[t] 或 T1[t] < T2[t]
    """
    P = np.asarray(prices, dtype=float)
    N = len(P)
    T1 = np.asarray(T1, dtype=float)
    T2 = np.asarray(T2, dtype=float)
    T3 = np.asarray(T3, dtype=float)

    position = np.zeros(N, dtype=int)  # 当天收盘结束后的持仓，用于“下一天”收益
    buy_idx = []
    sell_idx = []

    # 计算 RegimeLong
    RegimeLong = np.full(N, False, dtype=bool)
    for t in range(k_long, N):
        if np.isfinite(T1[t]) and np.isfinite(T2[t]) and np.isfinite(T3[t]) and np.isfinite(T3[t - k_long]):
            RegimeLong[t] = (T3[t] > T3[t - k_long]) and (T2[t] > T3[t]) and (T1[t] > T2[t])

    # 逐日更新仓位（t 日判定，t+1 生效）
    for t in range(max(k_long, 1), N - 1):
        # 默认延续
        position[t + 1] = position[t]

        # 需要数据可用
        if not (np.isfinite(T1[t]) and np.isfinite(T2[t]) and np.isfinite(T3[t])):
            continue
        if not (np.isfinite(T1[t - 1]) and np.isfinite(P[t - 1])):
            continue

        # 出场优先：环境失效 或 快线破位
        exit_cond = (not RegimeLong[t]) or (P[t] < T1[t]) or (T1[t] < T2[t])

        # 入场条件：多头环境 + 价格上穿快线
        entry_cond = RegimeLong[t] and (P[t - 1] <= T1[t - 1]) and (P[t] > T1[t])

        if exit_cond and position[t] == 1:
            position[t + 1] = 0
            sell_idx.append(t + 1)  # 执行日在 t+1

        # 只有空仓时才考虑入场
        if entry_cond and position[t] == 0:
            position[t + 1] = 1
            buy_idx.append(t + 1)   # 执行日在 t+1

    # 计算收益曲线（对数收益）
    log_ret = np.zeros(N)
    log_ret[1:] = np.log(P[1:] / P[:-1])
    strat_ret = np.zeros(N)
    strat_ret[1:] = position[:-1] * log_ret[1:]  # t 日持仓作用于 t+1 的收益

    equity = np.ones(N)
    equity[1:] = np.exp(np.cumsum(strat_ret[1:]))

    bh_equity = np.ones(N)
    bh_equity[1:] = np.exp(np.cumsum(log_ret[1:]))

    return {
        "position": position,
        "buy_idx": np.array(buy_idx, dtype=int),
        "sell_idx": np.array(sell_idx, dtype=int),
        "equity": equity,
        "bh_equity": bh_equity,
    }


# =========================
# 5. Performance
# =========================
def perf_report(equity, trading_days_per_year=252):
    equity = np.asarray(equity, dtype=float)
    total_return = equity[-1] / equity[0] - 1.0
    ret = equity[1:] / equity[:-1] - 1.0
    years = len(ret) / trading_days_per_year
    annual = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else np.nan
    return total_return, annual


# =========================
# 6. Plotting
# =========================
def plot_signals(prices, dates, buy_idx, sell_idx, title, save_path):
    dates = pd.to_datetime(dates)
    P = np.asarray(prices, dtype=float)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, P, label="CSI 300", linewidth=1.0)
    if buy_idx.size > 0:
        ax.scatter(dates[buy_idx], P[buy_idx], marker="^", s=40, label="Buy", alpha=0.8)
    if sell_idx.size > 0:
        ax.scatter(dates[sell_idx], P[sell_idx], marker="v", s=40, label="Sell", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Level")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved signals chart: {save_path}")


def plot_equity(dates, equity, bh_equity, title, save_path):
    dates = pd.to_datetime(dates)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(dates, bh_equity, label="Buy & Hold", linewidth=1.0)
    ax.plot(dates, equity, label="Strategy", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalized)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved equity chart: {save_path}")


# =========================
# 7. Main
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 载入数据
    close, dates = get_hs300_close(START_DATE, END_DATE)
    values = close.values
    N = len(values)

    # 预先计算所有需要的 drop 值（一次分解，服务多组配置）
    needed_drops = set()
    for f, m, s in DROP_TRIPLETS:
        assert f < m < s, "Require fast < mid < slow for drops."
        needed_drops.update([f, m, s])

    print("Computing rolling CEEMDAN trends for drops:", sorted(needed_drops))
    trend_map = compute_rolling_trends_lastpoint(values, WINDOW, needed_drops,
                                                 trials=TRIALS, epsilon=EPSILON, seed=SEED)

    # 逐组合回测、画图、输出报告
    for fast_d, mid_d, slow_d in DROP_TRIPLETS:
        T1 = trend_map[fast_d]
        T2 = trend_map[mid_d]
        T3 = trend_map[slow_d]

        res = run_strategy_long_flat(values, dates, T1, T2, T3, k_long=K_LONG)

        total, annual = perf_report(res["equity"])

        tag = f"d{fast_d}-{mid_d}-{slow_d}"
        print(f"\n=== Strategy Report [{tag}] ===")
        print(f"Total Return:   {total:.2%}")
        print(f"Annual Return:  {annual:.2%}")

        # 图1：价格 + 买卖点
        signals_path = os.path.join(OUTPUT_DIR, f"hs300_signals_{tag}.png")
        plot_signals(values, dates, res["buy_idx"], res["sell_idx"],
                     title=f"CSI 300 Signals (drop {fast_d},{mid_d},{slow_d})",
                     save_path=signals_path)

        # 图2：收益曲线
        equity_path = os.path.join(OUTPUT_DIR, f"hs300_equity_{tag}.png")
        plot_equity(dates, res["equity"], res["bh_equity"],
                    title=f"Equity Curve (drop {fast_d},{mid_d},{slow_d})",
                    save_path=equity_path)

    print("\nAll done.")


if __name__ == "__main__":
    main()
