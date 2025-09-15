# -*- coding: utf-8 -*-
"""
Per-station salinity plots (one Excel sheet per station).

Pipeline:
1) Parse datetime & salinity.
2) Daily mean -> 3-day moving average (centered). Missing days stay NaN.
3) Keep only Sep–Dec & Jan–Mar (across all years).
4) Overlays on the 3-day series:
   - Linear fitted trend line
   - 21-point moving-average smoothing (centered)
5) Missing values remain NaN so lines break over gaps (no drawing across missing).
6) X-axis ticks every 2 months; figure width scales with span.
7) All labels are ASCII-only (Chinese sheet names -> pinyin/ASCII).
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

# Optional pinyin conversion; falls back to ASCII-safe if not installed.
try:
    from pypinyin import lazy_pinyin  # pip install pypinyin
except Exception:
    lazy_pinyin = None

# ================== Paths (as requested) ==================
EXCEL_PATH = "/mnt/f/数据_东海局/salinity/盐度.xlsx"                 # path to your Excel
OUTPUT_DIR = Path("/mnt/f/wsl_plot/donghai/salinity/plots_salinity")     # output folder
# =========================================================

VALID_MONTHS = {9, 10, 11, 12, 1, 2, 3}  # Sep–Dec & Jan–Mar only


def to_pinyin_ascii(s: str) -> str:
    """Convert Chinese to pinyin, then keep ASCII-safe chars only."""
    s = "" if s is None else str(s)
    if lazy_pinyin is not None:
        try:
            s = " ".join(lazy_pinyin(s))
        except Exception:
            pass
    s = re.sub(r"[^A-Za-z0-9 _.\-]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or "sheet"


def sanitize_filename(s: str) -> str:
    s = to_pinyin_ascii(s)
    return re.sub(r"[\\/:\*\?\"<>\|]+", "_", s)


def find_datetime_column(df: pd.DataFrame) -> int:
    """Heuristically find datetime column index."""
    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]
    for key in ["日期", "时间", "datetime", "date", "time"]:
        if key in cols:
            return cols.index(key)
        if key in lower:
            return lower.index(key)
    for i in range(len(cols)):
        try:
            parsed = pd.to_datetime(df.iloc[:200, i], errors="coerce")
            if parsed.notna().mean() > 0.6:
                return i
        except Exception:
            pass
    return 0


def find_salinity_column(df: pd.DataFrame, dt_idx: int) -> int:
    """Heuristically find salinity column index."""
    cols = list(df.columns)
    lower = [str(c).strip().lower() for c in cols]
    for key in ["盐度", "salinity", "salt", "sal", "s"]:
        if key in cols:
            return cols.index(key)
        if key in lower:
            return lower.index(key)
    best = None
    for i in range(len(cols)):
        if i == dt_idx:
            continue
        ser = pd.to_numeric(df.iloc[:, i], errors="coerce")
        score = ser.notna().mean()
        if best is None or score > best[0]:
            best = (score, i)
    return best[1] if best else (1 if len(cols) > 1 else 0)


def month_span(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Inclusive month count between two timestamps."""
    return (end.year - start.year) * 12 + (end.month - start.month) + 1


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xldict = pd.read_excel(EXCEL_PATH, sheet_name=None)

    saved = []
    for sheet_name, df in xldict.items():
        # Pick columns
        dt_idx  = find_datetime_column(df)
        sal_idx = find_salinity_column(df, dt_idx)

        # Raw -> clean
        raw = pd.DataFrame({
            "datetime": pd.to_datetime(df.iloc[:, dt_idx], errors="coerce"),
            "salinity": pd.to_numeric(df.iloc[:, sal_idx], errors="coerce"),
        }).dropna(subset=["datetime", "salinity"])
        if raw.empty:
            print(f"[skip] {sheet_name}: no usable data")
            continue

        # ---- Daily mean (NaN for missing days so gaps remain) ----
        daily = (
            raw.set_index("datetime")
               .resample("D")
               .mean(numeric_only=True)           # NaN for days with no samples
        )  # DataFrame with index=DatetimeIndex, col='salinity'

        # Keep only requested months on the daily series
        daily = daily.loc[daily.index.month.isin(VALID_MONTHS)]
        if daily.empty:
            print(f"[skip] {sheet_name}: no data in requested months")
            continue

        # ---- 3-day moving average on daily series (centered) ----
        roll3 = daily["salinity"].rolling(window=3, center=True, min_periods=1).mean()
        mean3 = roll3.where(daily["salinity"].notna())  # keep NaN at missing days

        if np.isfinite(mean3.to_numpy()).sum() == 0:
            print(f"[skip] {sheet_name}: 3-day mean has no valid values")
            continue

        # Observed 3-day series to plot
        dt = mean3.index                      # DatetimeIndex
        y3 = mean3.to_numpy()

        # ---- Linear fit on 3-day series (use only valid points) ----
        valid_mask = np.isfinite(y3)
        if valid_mask.sum() >= 2:
            t0 = dt[valid_mask][0]
            t_days_full = ((dt - t0) / pd.Timedelta(days=1)).to_numpy(dtype=float)
            x_valid = t_days_full[valid_mask]
            y_valid = y3[valid_mask]
            slope, intercept = np.polyfit(x_valid, y_valid, 1)
            y_fit = np.full_like(y3, np.nan, dtype=float)
            y_fit[valid_mask] = slope * x_valid + intercept
        else:
            y_fit = np.full_like(y3, np.nan, dtype=float)

        # ---- 21-point smoothing on top of 3-day series ----
        orig = pd.Series(y3, index=dt)
        smooth21 = (
            orig.rolling(window=21, center=True, min_periods=1)
                .mean()
                .where(orig.notna())   # do not draw at missing days
                .to_numpy()
        )

        # ---- Dynamic width based on month span ----
        dmin = pd.to_datetime(daily.index.min())
        dmax = pd.to_datetime(daily.index.max())
        n_months = month_span(dmin, dmax)
        width = max(16, min(40, n_months * 0.6))  # widen to reduce crowding

        # ---- Plot (ASCII only) ----
        station_label = to_pinyin_ascii(sheet_name)
        fig, ax = plt.subplots(figsize=(width, 5.0), dpi=150)

        ax.plot(dt, y3, label="Observed (3-day mean)")
        ax.plot(dt, y_fit, label="Linear fit")
        ax.plot(dt, smooth21, label="Smoothed (21-pt MA)")

        ax.set_title(f"{station_label} - Salinity Time Series (Sep–Dec & Jan–Mar)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Salinity")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")

        # ---- X-axis ticks every 2 months ----
        ax.xaxis.set_major_locator(MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_horizontalalignment("right")

        fig.tight_layout()

        out_name = sanitize_filename(f"{station_label}_salinity_timeseries.png")
        out_path = OUTPUT_DIR / out_name
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)

        saved.append(str(out_path))
        print(f"[done] {sheet_name} -> {out_path}")

    # Index file
    index_path = OUTPUT_DIR / "outputs_index.txt"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write("\n".join(saved))
    print(f"\nAll done. Generated {len(saved)} figure(s).\nIndex: {index_path}\n")


if __name__ == "__main__":
    main()
