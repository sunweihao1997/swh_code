# -*- coding: utf-8 -*-
"""
Hourly -> Daily vector-mean wind & stick plot (one figure per sheet), EN labels.
Supports date filtering with --start/--end (inclusive) and shading intervals from a CSV.

CSV format expected (UTF-8/GBK auto):
    start_date,end_date[, ...]
Example:
    2022-09-14,2022-09-19
    2022-09-22,2022-10-08

Usage:
    python wind_daily_stickplot_en_range_shade.py \
        --input 风速风向.xlsx --plots out_dir \
        --start 2022-06-01 --end 2022-12-31 \
        --intervals ALL_sheets_exceed_intervals.csv --alpha 0.2
"""
import os
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# ----- NEW: optional pinyin support -----
try:
    from pypinyin import pinyin, Style  # optional dependency
    _HAS_PYPINYIN = True
except Exception:
    _HAS_PYPINYIN = False

PINYIN_MAP = {"佘山": "Sheshan", "崇明": "Chongming", "五好沟": "Wuhaogou"}

def _contains_cjk(s: str) -> bool:
    if not isinstance(s, str):
        s = str(s)
    return re.search(r"[\u4e00-\u9fff]", s) is not None

def to_ascii_pinyin(s: str) -> str:
    """
    Convert any Chinese characters in s to ASCII pinyin (no tones, TitleCase joined),
    keep ASCII characters unchanged. If pypinyin is unavailable, fall back to
    replacing non-ASCII with underscores so figures never show Hanzi.
    """
    if s is None:
        return ""
    s = str(s)
    # If we already have an explicit mapping, use it first (e.g., 佘山->Sheshan).
    mapped = PINYIN_MAP.get(s)
    if mapped:
        return mapped

    if _HAS_PYPINYIN and _contains_cjk(s):
        # Convert each Chinese character to its pinyin (no tone), title-case and join.
        parts = pinyin(s, style=Style.NORMAL, errors="ignore")  # list of lists
        py = "".join([seg[0].capitalize() for seg in parts if seg])
        # Keep ASCII letters/digits/space/hyphen/underscore from original s
        # around Chinese (e.g., "XX站A" -> "XXZhanA")
        # For simplicity we append the non-CJK ASCII characters as is.
        # Build a merged string: iterate original string and replace CJK with pinyin progressively
        out = []
        py_iter = iter(py)
        for ch in s:
            if re.match(r"[\u4e00-\u9fff]", ch):
                # Consume next piece from py_iter; we've already joined so just skip adding here
                # (The joined py already represents all CJK chars in order.)
                # We'll add nothing here and rely on 'py' at the end.
                pass
            else:
                out.append(ch)
        # Place pinyin at the beginning to ensure it's visible even if out is empty
        merged = py + "".join(out)
        # Finally, strip non-ASCII in case any slipped through
        merged = re.sub(r"[^\x00-\x7F]+", "_", merged)
        return merged if merged else py

    # Fallback: remove/replace non-ASCII
    ascii_only = re.sub(r"[^\x00-\x7F]+", "_", s)
    return ascii_only

def safe_filename(s: str) -> str:
    """Make a filesystem-safe ASCII filename fragment."""
    s = to_ascii_pinyin(s)
    s = re.sub(r"[^\w\-]+", "_", s).strip("_")
    return s or "site"

def load_intervals(csv_path):
    if not csv_path:
        return []
    try:
        idf = pd.read_csv(csv_path, encoding="utf-8", engine="python")
    except UnicodeDecodeError:
        idf = pd.read_csv(csv_path, encoding="gbk", engine="python")
    out = []
    for _, row in idf.iterrows():
        s = pd.to_datetime(row.get("start_date"), errors="coerce")
        e = pd.to_datetime(row.get("end_date"), errors="coerce")
        if pd.notna(s) and pd.notna(e):
            out.append((s, e + pd.Timedelta(days=1)))  # inclusive end
    return out

def infer_datetime_col(df: pd.DataFrame):
    candidates = []
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
        if parsed.notna().mean() > 0.6:
            candidates.append((parsed.notna().mean(), col))
    if not candidates:
        for c in ["datetime", "time", "date", "日期", "时间", "时刻", "时间戳"]:
            for col in df.columns:
                if c.lower() in str(col).lower():
                    return col
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]

def infer_speed_col(df: pd.DataFrame):
    priority = ["风速", "速度", "windspeed", "wind_speed", "ws", "speed", "平均风速"]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for key in priority:
        for low, orig in cols_lower.items():
            if key in low:
                if pd.api.types.is_numeric_dtype(df[orig]) or pd.to_numeric(df[orig], errors="coerce").notna().mean() > 0.6:
                    return orig
    for c in df.columns:
        low = str(c).lower()
        if any(u in low for u in ["m/s", "mps", "米每秒"]) and (
            pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.6
        ):
            return c
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8:
            return c
    return None

def infer_dir_col(df: pd.DataFrame):
    priority = ["风向", "方向", "winddir", "wind_dir", "wd", "dir", "direction"]
    cols_lower = {str(c).lower(): c for c in df.columns}
    for key in priority:
        for low, orig in cols_lower.items():
            if key in low:
                if pd.api.types.is_numeric_dtype(df[orig]) or pd.to_numeric(df[orig], errors="coerce").notna().mean() > 0.6:
                    return orig
    return None

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def wind_from_dir_speed_to_uv(direction_deg_from: pd.Series, speed: pd.Series):
    theta = np.deg2rad(direction_deg_from % 360.0)
    u = -speed * np.sin(theta)
    v = -speed * np.cos(theta)
    return u, v

def uv_to_speed_dir_from(u: pd.Series, v: pd.Series):
    speed = np.sqrt(u**2 + v**2)
    dir_rad = np.arctan2(-u, -v)
    direction_deg = (np.rad2deg(dir_rad) + 360.0) % 360.0
    return speed, direction_deg

def daily_vector_mean(df: pd.DataFrame, tcol: str, speed_col: str, dir_col: str) -> pd.DataFrame:
    work = df[[tcol, speed_col, dir_col]].copy()
    work[tcol] = pd.to_datetime(work[tcol], errors="coerce", infer_datetime_format=True)
    work = work.dropna(subset=[tcol]).sort_values(tcol).set_index(tcol)
    spd = to_numeric_safe(work[speed_col])
    direction = to_numeric_safe(work[dir_col])
    u, v = wind_from_dir_speed_to_uv(direction, spd)
    vec = pd.DataFrame({"U": u, "V": v}, index=work.index)
    daily_uv = vec.resample("D").mean()
    daily_speed, daily_dir = uv_to_speed_dir_from(daily_uv["U"], daily_uv["V"])
    out = pd.DataFrame({"U": daily_uv["U"], "V": daily_uv["V"], "Speed": daily_speed, "DirFrom": daily_dir}).dropna(how="all")
    return out

def make_stick_quiver(ax, dates_num, u, v, title_en: str, shade_intervals=None, alpha=0.2):
    import matplotlib.dates as mdates
    y = np.zeros_like(dates_num)
    max_speed = float(np.nanmax(np.sqrt(u**2 + v**2))) if len(u) else 1.0
    if not np.isfinite(max_speed) or max_speed == 0:
        max_speed = 1.0
    ylim = max(2.0, max_speed * 1.5)
    ax.set_ylim(-ylim, ylim)
    if shade_intervals:
        for (s, e) in shade_intervals:
            ax.axvspan(s, e, alpha=alpha, color="#FFA07A")  # use default color
    ax.quiver(dates_num, y, u, v, angles="xy", scale_units="xy", scale=None, pivot="middle",
              width=0.002, headwidth=5, headlength=6, headaxislength=5)
    ax.set_title(title_en)
    ax.set_ylabel("Vector magnitude (≈ m/s)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

def process_sheet(name: str, df: pd.DataFrame, plots_dir: str, start: pd.Timestamp, end: pd.Timestamp, shade_intervals, alpha: float):
    tcol = infer_datetime_col(df)
    spd_col = infer_speed_col(df)
    dir_col = infer_dir_col(df)
    if tcol is None or spd_col is None or dir_col is None:
        print(f"[WARN] sheet '{name}': unrecognized columns. time={tcol}, speed={spd_col}, dir={dir_col}.")
        return None

    daily = daily_vector_mean(df, tcol, spd_col, dir_col)
    daily = daily[(daily.index >= start) & (daily.index <= end)]
    if not daily.empty:
        import matplotlib.dates as mdates
        dates_num = mdates.date2num(daily.index.to_pydatetime())
        u = daily["U"].to_numpy(dtype=float)
        v = daily["V"].to_numpy(dtype=float)

        # ----- CHANGED: ensure no Hanzi in figure title or filename -----
        # 1) Map known names -> English; 2) convert remaining CJK to pinyin (ASCII).
        site_ascii = to_ascii_pinyin(PINYIN_MAP.get(name, name))

        plt.figure(figsize=(20, 5))
        ax = plt.gca()
        title_txt = f"{site_ascii}: Daily Mean Wind (Vector), {start.date()} to {end.date()}"
        make_stick_quiver(ax, dates_num, u, v, title_txt,
                          shade_intervals=shade_intervals, alpha=alpha)
        plt.tight_layout()
        os.makedirs(plots_dir, exist_ok=True)

        # Also sanitize the filename to stay ASCII-only
        fname_site = safe_filename(site_ascii)
        png_path = os.path.join(plots_dir, f"{fname_site}_{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}_daily_wind_shaded.pdf")
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        print(f"[INFO] sheet '{name}' has no data in given range.")
    return daily

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--plots", "-p", default="wind_daily_plots_en")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--intervals", type=str, default=None, help="CSV with columns start_date,end_date")
    ap.add_argument("--alpha", type=float, default=0.2, help="alpha for axvspan")
    args = ap.parse_args()
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    shade_intervals = load_intervals(args.intervals) if args.intervals else []
    xls = pd.ExcelFile(args.input)
    sheets = xls.sheet_names
    for sh in sheets:
        df = pd.read_excel(args.input, sheet_name=sh)
        process_sheet(sh, df, args.plots, start, end, shade_intervals, args.alpha)

if __name__ == "__main__":
    main()
