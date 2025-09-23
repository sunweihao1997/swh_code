"""
Diagnostic version (v5): plot EACH variable on its own figure for 2022 only.

Changes for diagnosis:
- DO NOT overlay. Plot 4 separate figures:
  1) Salinity (single series)
  2) Tide - Chongming
  3) Tide - Sheshan
  4) Tide - Wuhaogou
- Stronger numeric column detection to avoid picking station codes or ID columns.
- Keep figure labels in English/pinyin only (no Chinese).
- Fixed user paths embedded.

Outputs (PNG):
  /mnt/f/wsl_plot/donghai/salinity/salinity_2022.png
  /mnt/f/wsl_plot/donghai/salinity/tide_Chongming_2022.png
  /mnt/f/wsl_plot/donghai/salinity/tide_Sheshan_2022.png
  /mnt/f/wsl_plot/donghai/salinity/tide_Wuhaogou_2022.png

Dependencies:
  pip install pandas matplotlib openpyxl
"""

import os
from typing import Tuple, Dict, Optional
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Fixed paths (per user) ----------------
SALINITY_PATH = "/mnt/f/数据_东海局/salinity/qingcaosha_new.xlsx"
TIDE_PATH     = "/mnt/f/数据_东海局/salinity/潮位.xlsx"
OUT_DIR       = "/mnt/f/wsl_plot/donghai/salinity"

# If your salinity file has multiple sheets and you want to force one:
SALINITY_SHEET: Optional[str] = None

# Optional manual overrides for tide sheets (takes precedence if set)
STATION_SHEET_OVERRIDES: Dict[str, str] = {
    # "Sheshan": "06414佘山",
    # "Chongming": "05454崇明",
    # "Wuhaogou": "05518五好沟",
}

# Chinese -> Pinyin (for figure labels)
STATION_CN_TO_PY = {"佘山": "Sheshan", "崇明": "Chongming", "五好沟": "Wuhaogou"}
STATION_ALIASES = {
    "Sheshan": ["Sheshan", "She shan", "She-shan"],
    "Chongming": ["Chongming", "Chong ming", "Chong-ming"],
    "Wuhaogou": ["Wuhaogou", "Wu hao gou", "Wu-hao-gou", "Wuhao gou"],
}

# ---------------- Time utilities ----------------
def _normalize_time_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(idx)
    s = pd.to_datetime(s, errors="coerce")
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    # rounding to minute to reduce tiny misalignments (still fine for single-series plots)
    s = s.dt.round("T")
    return pd.DatetimeIndex(s)

# ---------------- Column detection ----------------
def _detect_time_and_value_columns_strong(df: pd.DataFrame) -> Tuple[str, str]:
    """
    Strong heuristic for (time_col, value_col):
    - Candidate time columns: any column with >=50% parseable datetimes (converted in-place).
    - Candidate numeric columns:
        * After coercion to numeric, require non-null ratio >= 0.3
        * Require unique count >= 20 (避免常数/枚举)
        * Use score = (#non-null) + log(variance + 1) 选最好的一列
    """
    df = df.copy()
    # find/convert datetime columns
    dt_cols = []
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
        if parsed.notna().mean() >= 0.5:
            df[c] = parsed
            dt_cols.append(c)
    if not dt_cols:
        raise ValueError("No parseable datetime column found.")
    time_col = dt_cols[0]

    # numeric candidates
    best_col, best_score = None, -np.inf
    for c in df.columns:
        if c == time_col:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        nonnull_ratio = vals.notna().mean()
        if nonnull_ratio < 0.30:
            continue
        nunique = pd.Series(vals.dropna().values).nunique()
        if nunique < 20:
            continue
        var = float(np.nanvar(vals))
        score = vals.notna().sum() + np.log1p(max(var, 0.0))
        if score > best_score:
            best_score = score
            best_col = c
    if best_col is None:
        # fallback: first numeric-ish column
        for c in df.columns:
            if c == time_col: 
                continue
            vals = pd.to_numeric(df[c], errors="coerce")
            if vals.notna().mean() >= 0.3:
                best_col = c
                break
    if best_col is None:
        raise ValueError("No suitable numeric data column found.")
    return time_col, best_col

def _df_to_series(df: pd.DataFrame, label: str) -> pd.Series:
    """Build a Series with normalized DatetimeIndex from a 2D DataFrame."""
    time_col, value_col = _detect_time_and_value_columns_strong(df)
    times = _normalize_time_index(pd.DatetimeIndex(df[time_col]))
    vals = pd.to_numeric(df[value_col], errors="coerce")
    s = pd.Series(vals.values, index=times, name=label)
    s = s[~s.index.isna()].dropna()
    s = s.groupby(s.index).mean().sort_index()
    return s

# ---------------- Loaders ----------------
def load_salinity_series(path: str, sheet: Optional[str]) -> pd.Series:
    if sheet:
        df = pd.read_excel(path, sheet_name=sheet)
        return _df_to_series(df, "Salinity")
    obj = pd.read_excel(path, sheet_name=None)
    if not isinstance(obj, dict):
        return _df_to_series(obj, "Salinity")
    # auto-pick best sheet
    best_name, best_score, best_series = None, -np.inf, None
    for nm, df in obj.items():
        try:
            ser = _df_to_series(df, "Salinity")
            score = ser.notna().sum()
        except Exception:
            ser, score = None, -1
        if ser is not None and score > best_score:
            best_name, best_score, best_series = nm, score, ser
    if best_series is None:
        raise ValueError("Failed to auto-detect salinity sheet.")
    print(f"[INFO] Auto-selected salinity sheet: {best_name} (points={best_score})")
    return best_series

def _normalize_name(s: str) -> str:
    return re.sub(r"[0-9\s_\-]+", "", s.lower())

def _find_sheet_fuzzy(available: list, target_cn: str, target_py: str, extra_aliases=None) -> Optional[str]:
    if target_cn in available:
        return target_cn
    for s in available:
        if target_cn in s:
            return s
    for s in available:
        if s.lower() == target_py.lower():
            return s
    for s in available:
        if target_py.lower() in s.lower():
            return s
    targets = {_normalize_name(target_py)}
    if extra_aliases:
        targets |= {_normalize_name(x) for x in extra_aliases}
    for s in available:
        ns = _normalize_name(s)
        if ns in targets or any(t in ns for t in targets):
            return s
    return None

def load_tide_series_dict(path: str) -> Dict[str, pd.Series]:
    xls = pd.ExcelFile(path)
    available = xls.sheet_names
    out = {}
    for cn, py in STATION_CN_TO_PY.items():
        if py in STATION_SHEET_OVERRIDES:
            sheet = STATION_SHEET_OVERRIDES[py]
            if sheet not in available:
                raise ValueError(f"Override sheet '{sheet}' for {py} not in {available}")
        else:
            sheet = _find_sheet_fuzzy(available, cn, py, STATION_ALIASES.get(py, []))
            if sheet is None:
                raise ValueError(f"Sheet for station '{py}' not found. Available: {available}")
        df = pd.read_excel(path, sheet_name=sheet)
        out[py] = _df_to_series(df, py)
        print(f"[INFO] Tide station '{py}' -> sheet '{sheet}', points={len(out[py])}")
    return out

# ---------------- Plot helper ----------------
def _plot_single(series: pd.Series, title: str, ylabel: str, outpath: str):
    s = series.copy()
    # keep only 2022
    s = s[(s.index >= pd.Timestamp("2022-01-01")) & (s.index <= pd.Timestamp("2022-12-31 23:59:59"))]
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(s.index, s.values, label=title)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[INFO] Saved: {outpath}  (n={len(s)})  range=[{np.nanmin(s.values) if len(s)>0 else 'NA'}, {np.nanmax(s.values) if len(s)>0 else 'NA'}]")

# ---------------- Main ----------------
def main():
    sal = load_salinity_series(SALINITY_PATH, SALINITY_SHEET)
    tides = load_tide_series_dict(TIDE_PATH)

    # Single-series figures (2022 only)
    _plot_single(sal, "Salinity - 2022", "Salinity", os.path.join(OUT_DIR, "salinity_2022.png"))
    for k in ["Chongming", "Sheshan", "Wuhaogou"]:
        _plot_single(tides[k], f"Tide - {k} - 2022", "Tide Level", os.path.join(OUT_DIR, f"tide_{k}_2022.png"))

if __name__ == "__main__":
    main()
