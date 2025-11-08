# -*- coding: utf-8 -*-
"""
2025-10-21
v6:
- Only process sheets whose names contain "宝钢"
- Output merged hourly & daily datasets (no per-sheet outputs)
- Plot one overlay chart: hourly (faint) + daily (bold, steps-mid), aligned in time
- No Chinese text in plotting (titles, labels, legends, filenames)
- Daily: resample to daily mean + full-date reindex + time interpolation
- Hourly: no interpolation, keep original hourly sampling
- Threshold shading and exceed-interval export based on daily
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pypinyin import lazy_pinyin

# === Defaults (adjust as needed) ===
EXCEL_PATH = "/mnt/f/数据_东海局/salinity/盐度 含宝钢水库.xlsx"
SHEET_KEYWORD = "宝钢"  # only sheets with this keyword in their names will be processed
OUTPUT_DIR = "/mnt/f/wsl_plot/donghai/salinity/baogang_daily_series"
DATE_COL_CANDIDATES = ["time", "datetime", "date", "时间", "日期", "采样时间"]

# —— Conversion & threshold —— #
CONVERT_FACTOR = 1.80655     # divisor
CONVERT_MULTIPLIER = 1000.0  # multiplier
THRESHOLD = 250.0            # threshold in converted units

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------- Utilities ----------
def to_pinyin(name):
    """Convert Chinese text to pinyin (no tones); spaces -> underscores; ensure ASCII-ish labels."""
    return "_".join(lazy_pinyin(str(name)))


def find_datetime_col(df: pd.DataFrame) -> str:
    """Heuristically find the datetime column."""
    for c in df.columns:
        if str(c).strip().lower() in DATE_COL_CANDIDATES or str(c).strip() in DATE_COL_CANDIDATES:
            return c
    first_col = df.columns[0]
    try:
        pd.to_datetime(df[first_col])
        return first_col
    except Exception:
        for c in df.columns:
            try:
                pd.to_datetime(df[c])
                return c
            except Exception:
                pass
    raise ValueError("Failed to detect a datetime column. Check DATE_COL_CANDIDATES or your input sheet.")


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build daily-mean series from hourly data:
      - set datetime index
      - resample to D (mean)
      - reindex to full daily span
      - time interpolation (internal gaps; ends remain NaN unless you ffill/bfill)
    """
    dt_col = find_datetime_col(df)
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(dt_col).sort_index()

    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for daily aggregation.")

    daily = df[num_cols].resample("D").mean()

    # Align to full daily index and interpolate missing days by time
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx)
    daily = daily.interpolate(method="time")
    # If you want to fill ends too, uncomment:
    # daily = daily.ffill().bfill()

    daily.index.name = "date"
    return daily


def make_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Extract original hourly numeric series (no aggregation, no interpolation)."""
    dt_col = find_datetime_col(df)
    df = df.copy()
    df[dt_col] = pd.to_datetime(df[dt_col])
    df = df.set_index(df[dt_col]).sort_index()
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise ValueError("No numeric columns found for hourly processing.")
    hourly = df[num_cols].copy()
    hourly.index.name = "datetime"
    return hourly


def slice_by_time_idxed(df_idxed: pd.DataFrame, start, end) -> pd.DataFrame:
    """Closed-interval slicing on a datetime-indexed DataFrame."""
    if start is None and end is None:
        return df_idxed
    if start is not None:
        start = pd.to_datetime(start)
    if end is not None:
        end = pd.to_datetime(end)
    if start is not None and end is not None:
        return df_idxed.loc[start:end]
    elif start is not None:
        return df_idxed.loc[start:]
    else:
        return df_idxed.loc[:end]


def convert_values(df: pd.DataFrame) -> pd.DataFrame:
    """Apply a global conversion: df / CONVERT_FACTOR * CONVERT_MULTIPLIER."""
    return df / CONVERT_FACTOR * CONVERT_MULTIPLIER


def shade_exceed_regions(ax, series_bool: pd.Series, alpha=0.10):
    """
    Shade vertical bands where the boolean series (indexed by datetime-like)
    is True in consecutive segments.
    """
    if series_bool.empty:
        return
    grp = (series_bool != series_bool.shift()).cumsum()
    for _, seg in series_bool.groupby(grp):
        if not seg.iloc[0]:
            continue
        start = seg.index[0]
        end = seg.index[-1]
        ax.axvspan(start, end, alpha=alpha, color="#FFA07A")


def bool_to_intervals(series_bool: pd.Series) -> pd.DataFrame:
    """
    Convert a daily boolean series (True=exceed) to intervals with start_date, end_date, duration_days.
    """
    out = []
    if series_bool.empty:
        return pd.DataFrame(columns=["start_date", "end_date", "duration_days"])

    grp = (series_bool != series_bool.shift()).cumsum()
    for _, seg in series_bool.groupby(grp):
        if not seg.iloc[0]:
            continue
        start = seg.index[0]
        end = seg.index[-1]
        duration_days = (end - start).days + 1
        out.append(
            {"start_date": start.date(), "end_date": end.date(), "duration_days": duration_days}
        )

    return pd.DataFrame(out)


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Baogang: hourly vs daily overlay (time window + daily interpolation + threshold shading + interval export)"
    )
    parser.add_argument("--excel", default=EXCEL_PATH, help="Path to Excel file")
    parser.add_argument("--keyword", default=SHEET_KEYWORD, help="Keyword to filter sheet names (default: 宝钢)")
    parser.add_argument("--outdir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--start", default=None, help="Start date/time inclusive, e.g., 2019-01-01")
    parser.add_argument("--end", default=None, help="End date/time inclusive, e.g., 2024-12-31")
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="Threshold in converted units")
    args = parser.parse_args()

    excel_path = args.excel
    keyword = args.keyword
    outdir = args.outdir
    start = args.start
    end = args.end
    threshold = args.threshold
    os.makedirs(outdir, exist_ok=True)

    # Read all sheets and keep only those whose names contain the keyword
    xls = pd.read_excel(excel_path, sheet_name=None, engine="openpyxl")
    target_items = [(name, df) for name, df in xls.items() if keyword in str(name)]
    if not target_items:
        raise ValueError(f"No sheets found with keyword '{keyword}' in their names.")

    hourly_list = []
    daily_list = []

    for name, df in target_items:
        # Hourly (no interpolation)
        hourly = make_hourly(df)
        hourly = convert_values(hourly)
        hourly = slice_by_time_idxed(hourly, start, end)

        # Daily (resample + full-date reindex + time interpolation)
        daily = to_daily(df)
        daily = convert_values(daily)
        daily = slice_by_time_idxed(daily, start, end)

        if hourly.empty or daily.empty:
            print(f"[Warning] Sheet '{name}' has no data in the selected window. Skipped.")
            continue

        pref = to_pinyin(name)
        hourly_list.append(hourly.add_prefix(f"{pref}_"))
        daily_list.append(daily.add_prefix(f"{pref}_"))

    if not hourly_list or not daily_list:
        raise ValueError("No usable data in the selected time window across the filtered sheets.")

    # Merge by columns (sheet prefix added to avoid collisions)
    hourly_all = pd.concat(hourly_list, axis=1).sort_index()
    daily_all = pd.concat(daily_list, axis=1).sort_index()

    # Export merged CSVs (converted values)
    hourly_out = os.path.join(outdir, "ALL_sheets_hourly_converted.csv")
    daily_out = os.path.join(outdir, "ALL_sheets_daily_converted.csv")
    hourly_all.to_csv(hourly_out, encoding="utf-8-sig")
    daily_all.to_csv(daily_out, encoding="utf-8-sig")

    # Exceed intervals based on daily (any column > threshold)
    exceed_any_all = (daily_all > threshold).any(axis=1)
    all_intervals = bool_to_intervals(exceed_any_all)
    all_intervals_path = os.path.join(outdir, "ALL_sheets_exceed_intervals.csv")
    all_intervals.to_csv(all_intervals_path, index=False, encoding="utf-8-sig")

    if all_intervals.empty:
        print(f"[ALL] No intervals exceeding threshold ({threshold:g}).")
    else:
        print(f"[ALL] Intervals exceeding threshold ({threshold:g}):")
        for _, r in all_intervals.iterrows():
            print(f"  {r['start_date']} ~ {r['end_date']}  (days: {int(r['duration_days'])})")

    # ===== Alignment for plotting =====
    # Shift daily index to midday (for plotting only) to visually center each day's mean
    daily_for_plot = daily_all.copy()
    daily_for_plot.index = daily_for_plot.index + pd.Timedelta(hours=12)

    # Use intersection window for plotting to avoid edge-only tails
    common_start = max(hourly_all.index.min(), daily_for_plot.index.min())
    common_end = min(hourly_all.index.max(), daily_for_plot.index.max())
    hourly_plot = hourly_all.loc[common_start:common_end]
    daily_plot = daily_for_plot.loc[common_start:common_end]

    # Prepare exceed boolean aligned to the shifted daily index for shading
    exceed_any_plot = exceed_any_all.copy()
    exceed_any_plot.index = exceed_any_plot.index + pd.Timedelta(hours=12)
    exceed_any_plot = exceed_any_plot.loc[common_start:common_end]

    # ===== Plot: hourly + daily overlay (ASCII only) =====
    fig, ax = plt.subplots(figsize=(20, 5))

    # 1) Plot hourly (thin, semi-transparent)
    for col in hourly_plot.columns:
        ax.plot(hourly_plot.index, hourly_plot[col], linewidth=0.8, alpha=0.5, label=None, color='k')

    # 2) Plot daily (bold, steps-mid)
    # Map common Chinese base names to English; others -> pinyin
    col_map = {"盐度": "Salinity", "温度": "Temperature"}
    shown_labels = set()
    for col in daily_plot.columns:
        # Expect "sheetprefix_basename", build label as "sheetprefix - basename"
        try:
            sheet_prefix, base = col.split("_", 1)
        except ValueError:
            sheet_prefix, base = "series", col
        base_label = col_map.get(str(base), "_".join(lazy_pinyin(str(base))))
        label = f"{sheet_prefix} - {base_label}"
        if label in shown_labels:
            label = f"{label} (daily)"
        shown_labels.add(label)

        ax.plot(
            daily_plot.index, daily_plot[col], color='red',
            linewidth=1.25, drawstyle="steps-mid", label=label
        )

    # Threshold line and shaded exceed regions (based on daily)
    ax.axhline(y=threshold, color='r', linestyle='--', linewidth=1.2, label=f"Threshold = {threshold:g}")
    shade_exceed_regions(ax, exceed_any_plot, alpha=0.10)

    # Axes & title (ASCII-only)
    time_suffix = f" [{common_start.strftime('%Y-%m-%d %H:%M')} ~ {common_end.strftime('%Y-%m-%d %H:%M')}]"
    ax.set_title(f"Baogang - Hourly (faint) vs Daily (bold, centered){time_suffix}", fontsize=14)
    ax.set_xlabel("Date / Time")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(loc="best", ncols=2, fontsize=8)
    fig.tight_layout()

    out_fig = os.path.join(outdir, "Baogang_overlay_hourly_daily_aligned.pdf")
    fig.savefig(out_fig, dpi=300)
    plt.close(fig)

    # Console summary
    print("Done.")
    print("  Hourly CSV:", os.path.abspath(hourly_out))
    print("  Daily  CSV:", os.path.abspath(daily_out))
    print("  Exceed intervals:", os.path.abspath(all_intervals_path))
    print("  Overlay plot:", os.path.abspath(out_fig))


if __name__ == "__main__":
    main()
