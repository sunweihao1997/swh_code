import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import akshare as ak
from PyEMD import CEEMDAN


# =========================
# 1. Data loader
# =========================

def get_hs300_close_from_sina_by_range(start_date=None, end_date=None, last_n=None):
    """
    获取沪深300指数日线收盘价（sh000300）

    参数:
        start_date: 字符串 'YYYY-MM-DD' 或 None
        end_date:   字符串 'YYYY-MM-DD' 或 None
        last_n:     在日期过滤后取最近 N 条记录（可为 None）

    返回:
        close: pd.Series(float)
        dates: pd.Series(datetime64)
    """
    df = ak.stock_zh_index_daily(symbol="sh000300")

    if df is None or df.empty:
        raise RuntimeError("Failed to fetch CSI 300 data from Sina. Check network or AkShare version.")

    # 标准化日期列
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

    # 日期过滤
    if start_date is not None:
        start_date = pd.to_datetime(start_date)
        df = df[df[date_col] >= start_date]
    if end_date is not None:
        end_date = pd.to_datetime(end_date)
        df = df[df[date_col] <= end_date]

    if df.empty:
        raise RuntimeError("No data after applying date filters.")

    # 过滤后截取最近 last_n 条
    if last_n is not None:
        if len(df) < last_n:
            raise RuntimeError(f"Not enough data after date filter ({len(df)} < {last_n})")
        df = df.tail(last_n)

    df = df.reset_index(drop=True)
    dates = df[date_col].copy()
    close = df["close"].astype(float).copy()

    print(f"Data range: {dates.iloc[0].date()} ~ {dates.iloc[-1].date()} ({len(close)} points)")
    return close, dates


# =========================
# 2. CEEMDAN decomposition
# =========================

def ceemdan_decompose(values, trials=100, epsilon=0.2, seed=42):
    """
    对一维数组执行 CEEMDAN 分解
    返回:
        imfs: (n_imf, N)
        residue: (N,)
    """
    values = np.asarray(values, dtype=float)

    ceemdan = CEEMDAN(trials=trials, epsilon=epsilon)
    ceemdan.noise_seed(seed)

    imfs = ceemdan.ceemdan(values)
    n_imf = imfs.shape[0]

    # 优先尝试库自带 residue
    residue = None
    try:
        imfs2, residue2 = ceemdan.get_imfs_and_residue()
        imfs2 = np.asarray(imfs2)
        residue2 = np.asarray(residue2)
        if imfs2.shape == imfs.shape:
            residue = residue2
    except AttributeError:
        pass

    # 回退：手算 residue
    if residue is None:
        residue = values - imfs.sum(axis=0)

    # 重构校验
    recon = imfs.sum(axis=0) + residue
    max_err = np.max(np.abs(values - recon))
    print(f"Number of IMFs: {n_imf}")
    print(f"Reconstruction error (max abs diff): {max_err:.3e}")

    return imfs, residue


# =========================
# 3. Reconstruction helper
# =========================

def reconstruct_from_imfs(imfs, residue, drop_first_n=0):
    """
    按 IMFs 重构平滑信号:
    - 丢弃前 drop_first_n 个 IMF，从 IMF(drop_first_n+1) 开始累加，再加 residue。
    - drop_first_n=0 时，相当于所有 IMF + residue（理论上≈原始信号）。
    """
    imfs = np.asarray(imfs, dtype=float)
    residue = np.asarray(residue, dtype=float)
    n_imf, _ = imfs.shape

    drop_first_n = max(0, int(drop_first_n))
    drop_first_n = min(drop_first_n, n_imf)

    if drop_first_n == n_imf:
        # 全部 IMF 被丢弃，只剩 residue
        return residue.copy()
    else:
        return imfs[drop_first_n:, :].sum(axis=0) + residue


# =========================
# 4. Plot functions
# =========================

def plot_all_imfs_with_dates(imfs, residue, dates, output_path):
    """
    绘制所有 IMF 分量 + residue（横轴为日期，每3个月刻度，带网格）
    """
    n_imf, N = imfs.shape
    assert len(dates) == N

    fig, axes = plt.subplots(n_imf + 1, 1,
                             figsize=(12, 2 * (n_imf + 1)),
                             sharex=True)

    if n_imf == 1:
        axes = [axes, ]

    for i in range(n_imf):
        axes[i].plot(dates, imfs[i], linewidth=0.8)
        axes[i].set_ylabel(f"IMF{i + 1}", fontsize=8)
        axes[i].grid(True, linestyle='--', alpha=0.5)

    axes[-1].plot(dates, residue, linewidth=0.8)
    axes[-1].set_ylabel("Residue", fontsize=8)
    axes[-1].grid(True, linestyle='--', alpha=0.5)

    axes[0].set_title("CSI 300 - CEEMDAN IMFs & Residue (Quarterly Ticks)")
    axes[-1].set_xlabel("Date")

    # 每3个月一个刻度
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"IMFs figure saved to: {output_path}")


def plot_trend_vs_original_from_imf4(imfs, residue, original_values, dates,
                                     drop_imf_n, output_path):
    """
    原始指数 vs (丢弃前 drop_imf_n 个 IMF，从下一个开始累加 + residue)
    用于展示“去掉高频后的趋势线”。
    """
    n_imf, N = imfs.shape
    assert len(dates) == N

    original_values = np.asarray(original_values, dtype=float)
    drop_imf_n = int(drop_imf_n)
    drop_imf_n = max(0, min(drop_imf_n, n_imf))

    if drop_imf_n >= n_imf:
        trend = residue.copy()
        label_trend = "Residue only"
    else:
        trend = imfs[drop_imf_n:, :].sum(axis=0) + residue
        label_trend = f"IMF{drop_imf_n + 1}+ ... +Residue (Drop IMF1-{drop_imf_n})"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(dates, original_values, label="Original CSI 300", linewidth=0.8)
    ax.plot(dates, trend, label=label_trend, linewidth=1.2)

    ax.set_title("CSI 300 - Original vs Reconstructed Trend (Drop High-frequency IMFs)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Level")

    # 日期刻度与网格
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Trend overlay figure saved to: {output_path}")


def plot_multi_trend_levels(original_values, dates, imfs, residue,
                            output_path=None):
    """
    绘制四条线，用于策略构思:
    1) 原始指数 original_values
    2) smooth_13: 丢弃 IMF1-3         -> 去高频
    3) smooth_14: 丢弃 IMF1-4         -> 去高频 + 一层中频
    4) smooth_15: 丢弃 IMF1-5         -> 去高频 + 更多中频（更慢趋势）
    """
    original_values = np.asarray(original_values, dtype=float)

    smooth_13 = reconstruct_from_imfs(imfs, residue, drop_first_n=3)
    smooth_14 = reconstruct_from_imfs(imfs, residue, drop_first_n=4)
    smooth_15 = reconstruct_from_imfs(imfs, residue, drop_first_n=5)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(dates, original_values, label="Original CSI 300", linewidth=0.8)
    ax.plot(dates, smooth_13,
            label="Drop IMF1-3",
            linewidth=1.2)
    ax.plot(dates, smooth_14,
            label="Drop IMF1-4",
            linewidth=1.4)
    ax.plot(dates, smooth_15,
            label="Drop IMF1-5",
            linewidth=1.6)

    ax.set_title("CSI 300 - Multi-level CEEMDAN Trends (Drop IMF1-3 / 1-4 / 1-5)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Level")

    # 日期刻度 & 网格
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend()
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200)
        print(f"Multi-level trend figure saved to: {output_path}")

    plt.close(fig)


# =========================
# 5. Main
# =========================

def main():
    output_dir = "/home/sun/paint/stock/"
    os.makedirs(output_dir, exist_ok=True)

    # 选择分析区间
    start_date = "2021-01-01"
    end_date = None
    last_n = None

    # 1) 获取数据
    close, dates = get_hs300_close_from_sina_by_range(
        start_date=start_date,
        end_date=end_date,
        last_n=last_n
    )
    values = close.values

    # 2) CEEMDAN 分解
    imfs, residue = ceemdan_decompose(values)

    # 3) 图一：所有 IMF + Residue
    imf_fig_path = os.path.join(output_dir, "csi300_ceemdan_imfs_date.png")
    plot_all_imfs_with_dates(imfs, residue, dates, imf_fig_path)

    # 4) 图二：原始 vs 丢弃 IMF1-3 的重构趋势
    trend_overlay_path = os.path.join(output_dir, "csi300_trend_drop_imf1_3_vs_original.png")
    plot_trend_vs_original_from_imf4(
        imfs=imfs,
        residue=residue,
        original_values=values,
        dates=dates,
        drop_imf_n=3,
        output_path=trend_overlay_path
    )

    # 5) 图三：原始 + 丢弃1-3 / 1-4 / 1-5 的三条平滑线
    multi_trend_path = os.path.join(output_dir, "csi300_multi_trend_levels_1_3_4_5.png")
    plot_multi_trend_levels(
        original_values=values,
        dates=dates,
        imfs=imfs,
        residue=residue,
        output_path=multi_trend_path
    )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
