# -*- coding: utf-8 -*-
"""
This script plots tide levels between 2022-08-01 and 2022-12-31 for the first 3 sheets of an Excel file,
and overlays exceed intervals using axvspan. All plot text (titles, axis labels) uses Pinyin/ASCII to avoid
Chinese characters in figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
from pathlib import Path

# --- Paths (keep your originals) ---
excel_path = "/mnt/f/data_donghai/salinity/潮位.xlsx"  # your file path
exceed_csv_path = "/mnt/f/wsl_plot/donghai/salinity/baogang_daily_series/ALL_sheets_exceed_intervals.csv"  # your file path

# --- Time window ---
start = pd.to_datetime("2024-09-15")
end = pd.to_datetime("2025-04-15")

# --- Pinyin mapping helpers (ASCII only for plots) ---
CN_PINYIN_MAP = {
    # common column names
    "潮位": "Chaowei",
    "时间": "Shijian",
    "日期": "Riqi",
    # known station names in sheet titles
    "崇明": "Chongming",
    "佘山": "Sheshan",
    "五好沟": "Wuhaogou",
}

def to_pinyin_ascii(s: str) -> str:
    """Replace known Chinese tokens with Pinyin and strip remaining non-ASCII to underscores."""
    out = str(s)
    for cn, py in CN_PINYIN_MAP.items():
        out = out.replace(cn, py)
    # ensure ASCII only
    out = "".join(ch if ord(ch) < 128 else "_" for ch in out)
    # collapse duplicate underscores
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_ ")

# --- Read exceed intervals ---
exceed_df = pd.read_csv(exceed_csv_path)
exceed_df = exceed_df.rename(columns={
    "start": "start_date",
    "Start": "start_date",
    "start time": "start_date",
    "end": "end_date",
    "End": "end_date",
    "end time": "end_date"
})
for col in ["start_date", "end_date"]:
    if col in exceed_df.columns:
        exceed_df[col] = pd.to_datetime(exceed_df[col])
    else:
        raise ValueError(f"Required column not found in {exceed_csv_path}: {col}")

# --- Read Excel and process first 3 sheets ---
xl = pd.ExcelFile(excel_path)
sheet_names = xl.sheet_names[:4]  # if you truly want only first 3, change to [:3]

time_candidates = ["日期", "时间", "datetime", "Datetime", "date", "Date", "时间戳", "timestamp"]
level_candidates = ["潮位", "水位", "water level", "level", "Level"]

for name in sheet_names:
    df = xl.parse(name)

    # auto-detect time and level columns
    time_col = next((c for c in df.columns if str(c).strip() in time_candidates), None)
    level_col = next((c for c in df.columns if str(c).strip() in level_candidates), None)

    if time_col is None or level_col is None:
        print(f"[{name}] missing required columns (time: {time_candidates}; level: {level_candidates}), skip.")
        continue

    df = df[[time_col, level_col]].copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col])

    # filter by time window
    mask = (df[time_col] >= start) & (df[time_col] <= end)
    dff = df[mask].dropna(subset=[level_col]).sort_values(by=time_col)

    if dff.empty:
        print(f"[{name}] no data in {start.date()} to {end.date()}, skip plotting.")
        continue

    # --- Build ASCII/pinyin labels for plotting ---
    sheet_display = to_pinyin_ascii(name)
    x_label = "Shijian"    # pinyin for 时间
    y_label = "Chaowei"    # pinyin for 潮位
    title = f"{sheet_display}: Chaowei ({start.date()} ~ {end.date()})"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(dff[time_col], dff[level_col], linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # x-axis formatter
    locator = AutoDateLocator()
    formatter = DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    # overlay exceed intervals intersecting with window
    for _, row in exceed_df.iterrows():
        s = row["start_date"]
        e = row["end_date"]
        if pd.isna(s) or pd.isna(e):
            continue
        left = max(start, s)
        right = min(end, e)
        if left <= right:
            ax.axvspan(left, right, alpha=0.2, color="#FFA07A")

    # save figure; use pinyin-only basename
    safe_name = to_pinyin_ascii(name)
    out_path = Path(f"/mnt/f/wsl_plot/donghai/salinity/{safe_name}_tide_20240915_20250415.pdf")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path.resolve()}")

print("Done.")
