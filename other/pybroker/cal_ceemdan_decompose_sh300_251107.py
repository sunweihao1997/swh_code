import os
import akshare as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyEMD import CEEMDAN


# =========================
# 1. Data loader (Sina via AkShare)
# =========================

def get_hs300_close_from_sina(last_n=1000):
    """
    Fetch CSI 300 index daily close prices using Sina data via AkShare.
    """
    df = ak.stock_zh_index_daily(symbol="sh000300")

    if df is None or df.empty:
        raise RuntimeError("Failed to fetch CSI 300 data from Sina. Check network or AkShare version.")

    # Ensure sorted by date
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
        dates = df["date"]
    else:
        df = df.sort_index()
        dates = df.index.to_series()

    if "close" not in df.columns:
        raise RuntimeError(f"'close' column not found. Actual columns: {df.columns}")

    if len(df) < last_n:
        raise RuntimeError(f"Not enough data ({len(df)} < {last_n})")

    sub = df.tail(last_n).copy()
    close = sub["close"].astype(float).reset_index(drop=True)
    dates = dates.tail(last_n).reset_index(drop=True)

    print(f"Data range: {dates.iloc[0]} ~ {dates.iloc[-1]} ({len(close)} points)")
    return close, dates


# =========================
# 2. CEEMDAN decomposition
# =========================

def ceemdan_decompose(values, trials=100, epsilon=0.2, seed=42):
    """
    Run CEEMDAN on a 1D numpy array.
    Returns: imfs (n_imf, N), residue (N,)
    """
    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)
    ceemdan.noise_seed(seed)

    imfs = ceemdan.ceemdan(values)
    n_imf = imfs.shape[0]

    # Try to get residue from library if available, else compute manually.
    residue = None
    try:
        imfs2, residue2 = ceemdan.get_imfs_and_residue()
        if imfs2.shape == imfs.shape:
            residue = residue2
    except AttributeError:
        pass

    if residue is None:
        residue = values - imfs.sum(axis=0)

    recon = imfs.sum(axis=0) + residue
    max_err = np.max(np.abs(values - recon))
    print(f"Number of IMFs: {n_imf}")
    print(f"Reconstruction error (max abs diff): {max_err:.3e}")

    return imfs, residue


# =========================
# 3. Aggregate components
# =========================

def aggregate_components(imfs, residue, hf_imf_count=3, trend_imf_from=4):
    """
    Aggregate IMFs into:
    - high-frequency component: IMF1..IMF_hf_imf_count
    - trend component: IMF_trend_imf_from..IMF_n + residue
    """
    n_imf, _ = imfs.shape

    # High-frequency
    hf_end = min(hf_imf_count, n_imf)
    hf_part = imfs[0:hf_end, :].sum(axis=0)

    # Trend
    if trend_imf_from <= n_imf:
        trend_part = imfs[trend_imf_from - 1:, :].sum(axis=0) + residue
    else:
        # Fallback: if config exceeds, approximate trend as total - high-frequency
        trend_part = imfs.sum(axis=0) + residue - hf_part

    return hf_part, trend_part


# =========================
# 4. Timing signal (toy example)
# =========================

def build_timing_signal(prices, trend_part, ma_window=20, slope_lookback=3):
    """
    Very simple example timing rule based on CEEMDAN trend:

    - Long (1) when trend > MA(trend) and trend is rising.
    - Short (-1) when trend < MA(trend) and trend is falling.
    - Else 0.

    Uses previous day's signal on today's return (no look-ahead).

    Returns:
        signal: (N,) in {1, 0, -1}
        equity: (N,) strategy equity curve normalized to 1
    """
    prices = np.asarray(prices)
    N = len(prices)

    trend = pd.Series(trend_part)
    trend_ma = trend.rolling(ma_window).mean()
    slope = trend.diff(slope_lookback)

    signal = np.zeros(N)

    for i in range(max(ma_window, slope_lookback), N):
        if trend[i] > trend_ma[i] and slope[i] > 0:
            signal[i] = 1
        elif trend[i] < trend_ma[i] and slope[i] < 0:
            signal[i] = -1
        else:
            signal[i] = 0

    # Daily log returns of index
    log_ret = np.zeros(N)
    log_ret[1:] = np.log(prices[1:] / prices[:-1])

    # Strategy return uses previous day's signal
    strat_ret = signal[:-1] * log_ret[1:]

    # Build equity curve, same length N, starting at 1
    equity = np.ones(N)
    equity[1:] = np.exp(np.cumsum(strat_ret))

    return signal, equity


# =========================
# 5. Performance metrics
# =========================

def compute_performance(equity, trading_days_per_year=252):
    """
    Compute performance metrics:
    - total return
    - annualized return
    - Sharpe ratio (zero risk-free)
    - max drawdown
    """
    equity = np.asarray(equity, dtype=float)
    if len(equity) < 2:
        return {
            "total_return": np.nan,
            "annual_return": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
        }

    # Daily returns
    ret = equity[1:] / equity[:-1] - 1

    total_return = equity[-1] / equity[0] - 1

    years = len(ret) / trading_days_per_year
    if years > 0:
        annual_return = (1.0 + total_return) ** (1.0 / years) - 1.0
    else:
        annual_return = np.nan

    # Sharpe ratio
    std = ret.std(ddof=1)
    if std > 0:
        sharpe = (ret.mean() * np.sqrt(trading_days_per_year)) / std
    else:
        sharpe = np.nan

    # Max drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_drawdown = drawdown.min()

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def print_performance(name, perf):
    """
    Pretty-print performance dictionary.
    """
    tr = perf["total_return"]
    ar = perf["annual_return"]
    sh = perf["sharpe"]
    mdd = perf["max_drawdown"]

    def fmt(x):
        return "NA" if x is None or np.isnan(x) else f"{x:.2%}"

    print(f"\n=== {name} Performance ===")
    print(f"Total return:      {fmt(tr)}")
    print(f"Annualized return: {fmt(ar)}")
    print(f"Sharpe ratio:      {'NA' if np.isnan(sh) else f'{sh:.2f}'}")
    print(f"Max drawdown:      {fmt(mdd)}")


# =========================
# 6. Main
# =========================

def main():
    output_dir = "/home/sun/paint/stock/"
    os.makedirs(output_dir, exist_ok=True)

    # 1) Fetch data
    close, dates = get_hs300_close_from_sina(last_n=1000)
    values = close.values
    t = np.arange(len(values))

    # 2) CEEMDAN
    imfs, residue = ceemdan_decompose(values)

    # 3) Aggregate components
    # IMF1–3: high-frequency; IMF4+ & residue: trend (based on your previous decomposition shape)
    hf_part, trend_part = aggregate_components(imfs, residue,
                                               hf_imf_count=3,
                                               trend_imf_from=4)

    # 4) Build timing strategy
    signal, strat_equity = build_timing_signal(values, trend_part,
                                               ma_window=20,
                                               slope_lookback=3)

    # 5) Buy & Hold equity (CSI 300 normalized)
    bh_equity = values / values[0]

    # 6) Compute performance
    perf_bh = compute_performance(bh_equity)
    perf_strat = compute_performance(strat_equity)

    print_performance("Buy & Hold (CSI 300)", perf_bh)
    print_performance("CEEMDAN Timing Strategy", perf_strat)

    # =========================
    # Plot 1: Original vs Trend
    # =========================
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(t, values, label="Original close", linewidth=1.0)
    ax1.plot(t, trend_part, label="Reconstructed trend", linewidth=1.5)
    ax1.set_title("CSI 300 - Original vs CEEMDAN-based trend")
    ax1.set_xlabel("Sample index (time order)")
    ax1.set_ylabel("Index level")
    ax1.legend()
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, "csi300_trend.png"), dpi=200)
    plt.show()

    # =========================
    # Plot 2: High-frequency component
    # =========================
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(t, hf_part, linewidth=0.8)
    ax2.set_title("CSI 300 - High-frequency component (IMF1–IMF3)")
    ax2.set_xlabel("Sample index (time order)")
    ax2.set_ylabel("Value")
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, "csi300_highfreq.png"), dpi=200)
    plt.show()

    # =========================
    # Plot 3: Timing signal on price
    # =========================
    fig3, ax3 = plt.subplots(figsize=(12, 5))
    ax3.plot(t, values, label="Original close", linewidth=1.0)
    long_idx = np.where(signal == 1)[0]
    short_idx = np.where(signal == -1)[0]
    ax3.scatter(long_idx, values[long_idx], marker="^", s=20,
                label="Long", alpha=0.7)
    ax3.scatter(short_idx, values[short_idx], marker="v", s=20,
                label="Short", alpha=0.7)
    ax3.set_title("CSI 300 - CEEMDAN-based timing signal")
    ax3.set_xlabel("Sample index (time order)")
    ax3.set_ylabel("Index level")
    ax3.legend()
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, "csi300_signal.png"), dpi=200)
    plt.show()

    # =========================
    # Plot 4: Strategy equity curve (alone)
    # =========================
    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(t, strat_equity, linewidth=1.0)
    ax4.set_title("CEEMDAN timing strategy - equity curve")
    ax4.set_xlabel("Sample index (time order)")
    ax4.set_ylabel("Equity (normalized)")
    plt.tight_layout()
    fig4.savefig(os.path.join(output_dir, "csi300_equity_strategy.png"), dpi=200)
    plt.show()

    # =========================
    # Plot 5: Strategy vs Buy & Hold
    # =========================
    fig5, ax5 = plt.subplots(figsize=(12, 4))
    ax5.plot(t, bh_equity, label="Buy & Hold (CSI 300)", linewidth=1.0)
    ax5.plot(t, strat_equity, label="Timing strategy", linewidth=1.2)
    ax5.set_title("CSI 300 - Timing strategy vs Buy & Hold")
    ax5.set_xlabel("Sample index (time order)")
    ax5.set_ylabel("Equity (normalized)")
    ax5.legend()
    plt.tight_layout()
    fig5.savefig(os.path.join(output_dir, "csi300_equity_timing_vs_bh.png"), dpi=200)
    plt.show()

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
