"""
Final script (robust v4): Plot 2022 salinity vs. tide (3 stations) and compute correlations.

Updates:
- Embed fixed paths per user request.
- Robust time alignment: strip tz, round to nearest 1 minute, deduplicate by index mean, THEN intersect.
- Fuzzy matching for tide sheet names like '06414佘山', etc.
- Auto-detect salinity sheet if multiple; can also force one.
- One figure, English/pinyin only; correlations shown on-plot.

Outputs:
- Figure: /mnt/f/wsl_plot/donghai/salinity/salinity_tide_2022.png
- Console: counts before/after alignment + correlations.

Dependencies:
    pip install pandas matplotlib openpyxl
"""

import os
from typing import Tuple, Dict, Optional
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ====== Fixed paths (per user) ======
SALINITY_PATH = "/mnt/f/数据_东海局/salinity/qingcaosha_new.xlsx"
TIDE_PATH     = "/mnt/f/数据_东海局/salinity/潮位.xlsx"
OUT_FIG_PATH  = "/mnt/f/wsl_plot/donghai/salinity/salinity_tide_2022.png"

# If your salinity file has multiple sheets and you want to force one, set this:
# e.g. SALINITY_SHEET = "Sheet1"  or  "salt_2022"  or  "盐度"
SALINITY_SHEET: Optional[str] = None

# Optional: manually override station -> sheet name if needed (takes precedence)
# Example (uncomment if you want to force):
# STATION_SHEET_OVERRIDES = {
#     "Sheshan": "06414佘山",
#     "Chongming": "05454崇明",
#     "Wuhaogou": "05518五好沟",
# }
STATION_SHEET_OVERRIDES: Dict[str, str] = {}

# Map Chinese station names to pinyin labels (for figure labels with no Chinese)
STATION_CN_TO_PY = {
    "佘山": "Sheshan",
    "崇明": "Chongming",
    "五好沟": "Wuhaogou",
}

# Also accept some alias spellings (in case of variants)
STATION_ALIASES = {
    "Sheshan": ["Sheshan", "She shan", "She-shan"],
    "Chongming": ["Chongming", "Chong ming", "Chong-ming"],
    "Wuhaogou": ["Wuhaogou", "Wu hao gou", "Wu-hao-gou", "Wuhao gou"],
}

# -------------------------------- Utility --------------------------------

def _strip_tz(ts: pd.Series) -> pd.Series:
    """Remove timezone info (convert to naive) if present."""
    if hasattr(ts.dt, "tz") and ts.dt.tz is not None:
        return ts.dt.tz_convert(None)
    return ts

def _normalize_time_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Normalize timestamps to improve alignment:
    - Convert to naive (no tz) if tz-aware.
    - Round to nearest 1 minute (avoid second/millisecond mismatches).
    """
    s = pd.Series(idx)
    # ensure datetime dtype
    s = pd.to_datetime(s, errors="coerce")
    # strip tz if any
    if hasattr(s.dt, "tz") and s.dt.tz is not None:
        s = s.dt.tz_convert(None)
    # round to nearest minute
    s = s.dt.round("T")
    return pd.DatetimeIndex(s)

def _detect_time_and_value_columns(df: pd.DataFrame) -> Tuple[str, str]:
    """Heuristically detect the datetime column and one numeric column."""
    # Try to parse each column as datetime (don't modify original df)
    dt_cols = []
    for c in df.columns:
        parsed = pd.to_datetime(df[c], errors="coerce", utc=False, infer_datetime_format=True)
        if parsed.notna().mean() >= 0.5:
            dt_cols.append(c)
            # keep parsed in-place for later use
            df[c] = parsed
    if not dt_cols:
        raise ValueError("No parseable datetime column found.")
    time_col = dt_cols[0]

    # Numeric column
    numeric_candidates = [c for c in df.columns if c != time_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_candidates:
        for c in df.columns:
            if c == time_col:
                continue
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() >= 0.5:
                df[c] = coerced
                numeric_candidates.append(c)
                break
    if not numeric_candidates:
        raise ValueError("No usable numeric column found.")

    value_col = numeric_candidates[0]
    return time_col, value_col

def _df_to_series(df: pd.DataFrame, label: str) -> pd.Series:
    """Convert a 2D DataFrame to a single Series with normalized DatetimeIndex."""
    df = df.copy()
    time_col, value_col = _detect_time_and_value_columns(df)

    times = pd.to_datetime(df[time_col], errors="coerce")
    # normalize time index
    times = _normalize_time_index(pd.DatetimeIndex(times))
    vals = pd.to_numeric(df[value_col], errors="coerce")

    s = pd.Series(vals.values, index=times)
    s = s[~s.index.isna()]
    s = s.dropna()
    # remove duplicate timestamps by keeping mean
    s = s.groupby(s.index).mean()
    s = s.sort_index()
    s.name = label
    return s

# ----------------------------- Load Functions -----------------------------

def load_single_series_from_excel(path: str, sheet_name=None, label: str = None) -> pd.Series:
    """Load a single time series from an Excel sheet/file; auto-detect the best sheet if needed."""
    if sheet_name is not None:
        df = pd.read_excel(path, sheet_name=sheet_name)
        if isinstance(df, dict):
            raise ValueError(f"Sheet '{sheet_name}' not found as a single DataFrame.")
        return _df_to_series(df, label or "value")

    obj = pd.read_excel(path, sheet_name=None)  # dict of {sheet: DataFrame} OR a DataFrame
    if not isinstance(obj, dict):
        return _df_to_series(obj, label or "value")

    best_name, best_score, best_series = None, -np.inf, None
    for nm, df in obj.items():
        try:
            ser = _df_to_series(df, label or "value")
            score = ser.notna().sum()
        except Exception:
            score = -1
            ser = None
        if score > best_score and ser is not None:
            best_name, best_score, best_series = nm, score, ser
    if best_series is None:
        raise ValueError("Failed to auto-detect a valid sheet for salinity. "
                         "Please set SALINITY_SHEET to the correct sheet name.")
    print(f"[INFO] Auto-selected salinity sheet: {best_name} (points={best_score})")
    best_series.name = label or "Salinity"
    return best_series

def _normalize_name(s: str) -> str:
    """Lowercase and remove digits/whitespace/_-/ to help fuzzy matching."""
    s = s.lower()
    s = re.sub(r"[0-9\s_\-]+", "", s)
    return s

def _find_sheet_fuzzy(available: list, target_cn: str, target_py: str, extra_aliases=None) -> Optional[str]:
    """Fuzzy find a sheet name among available ones."""
    # exact Chinese
    if target_cn in available:
        return target_cn
    # contains Chinese
    for s in available:
        if target_cn in s:
            return s
    # equality to pinyin (case-insensitive)
    for s in available:
        if s.lower() == target_py.lower():
            return s
    # substring pinyin
    for s in available:
        if target_py.lower() in s.lower():
            return s
    # normalized match
    norm_targets = { _normalize_name(target_py) }
    if extra_aliases:
        norm_targets.update({ _normalize_name(x) for x in extra_aliases })
    for s in available:
        ns = _normalize_name(s)
        if ns in norm_targets or any(nt in ns for nt in norm_targets):
            return s
    return None

def load_tide_three_stations(path: str) -> Dict[str, pd.Series]:
    """Load tide data from 3 stations using robust fuzzy sheet matching."""
    xls = pd.ExcelFile(path)
    available_sheets = xls.sheet_names

    stations = {}
    for cn, py in STATION_CN_TO_PY.items():
        # manual override first
        if py in STATION_SHEET_OVERRIDES:
            sheet = STATION_SHEET_OVERRIDES[py]
            if sheet not in available_sheets:
                raise ValueError(f"Override sheet '{sheet}' for station '{py}' not found. Available: {available_sheets}")
        else:
            sheet = _find_sheet_fuzzy(
                available_sheets, target_cn=cn, target_py=py,
                extra_aliases=STATION_ALIASES.get(py, [])
            )
            if sheet is None:
                raise ValueError(f"Sheet for station '{py}' not found. Available: {available_sheets}")

        df = pd.read_excel(path, sheet_name=sheet)
        stations[py] = _df_to_series(df, py)

    return stations

# ----------------------------- Alignment & Calc ---------------------------

def align_common_2022(salin: pd.Series, tides: Dict[str, pd.Series]) -> Tuple[pd.Series, Dict[str, pd.Series]]:
    """Keep only timestamps common to salinity and ALL tide series within 2022; drop NaNs."""
    start = pd.Timestamp("2022-01-01 00:00:00")
    end   = pd.Timestamp("2022-12-31 23:59:59")

    sal_2022 = salin[(salin.index >= start) & (salin.index <= end)].copy()
    tides_2022 = {k: v[(v.index >= start) & (v.index <= end)].copy() for k, v in tides.items()}

    # diagnostics
    print(f"[INFO] Points in 2022 — Salinity: {len(sal_2022)}; "
          + "; ".join([f"{k}: {len(tides_2022[k])}" for k in sorted(tides_2022.keys())]))

    # strict intersection of all indices
    common_index = sal_2022.index
    for v in tides_2022.values():
        common_index = common_index.intersection(v.index)

    print(f"[INFO] Common timestamps after rounding (strict intersection): {len(common_index)}")

    sal_2022 = sal_2022.reindex(common_index)
    tides_2022 = {k: v.reindex(common_index) for k, v in tides_2022.items()}

    df = pd.concat([sal_2022.rename("Salinity")] + [tides_2022[k].rename(k) for k in sorted(tides_2022.keys())], axis=1)
    before_drop = len(df)
    df = df.dropna(how="any")
    print(f"[INFO] Rows before dropna: {before_drop}; after dropna (aligned & complete): {len(df)}")

    sal_clean = df["Salinity"]
    tides_clean = {k: df[k] for k in sorted(tides_2022.keys())}
    return sal_clean, tides_clean

def compute_correlations(sal: pd.Series, tides: Dict[str, pd.Series]) -> Dict[str, float]:
    """Pearson correlation between salinity and each tide station series."""
    corrs = {}
    for k, v in tides.items():
        corrs[k] = sal.corr(v) if len(sal) > 1 and len(v) > 1 else np.nan
    return corrs

# --------------------------------- Plot -----------------------------------

def make_figure(sal: pd.Series, tides: Dict[str, pd.Series], corrs: Dict[str, float], outpath: str):
    """Single figure: left y for salinity, right y for 3 tide series; English/pinyin only."""
    plt.figure(figsize=(12, 6))
    ax1 = plt.gca()
    l1, = ax1.plot(sal.index, sal.values, label="Salinity")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Salinity")

    ax2 = ax1.twinx()
    lines2 = []
    for k in sorted(tides.keys()):
        line, = ax2.plot(tides[k].index, tides[k].values, label=f"Tide - {k}")
        lines2.append(line)
    ax2.set_ylabel("Tide Level")

    lines = [l1] + lines2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_title("Salinity and Tide (Sheshan / Chongming / Wuhaogou) - 2022")

    corr_text_lines = [f"Corr(Salinity, {k}) = {corrs[k]:.3f}" if pd.notna(corrs[k]) else f"Corr(Salinity, {k}) = NaN"
                       for k in sorted(tides.keys())]
    corr_text = "\n".join(corr_text_lines)
    ax1.text(0.99, 0.98, corr_text, transform=ax1.transAxes, ha="right", va="top",
             bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    plt.tight_layout()
    # Ensure output directory exists
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[INFO] Figure saved to: {outpath}")

# --------------------------------- Main -----------------------------------

def main():
    # Load salinity (auto-detect best sheet unless SALINITY_SHEET is set)
    if SALINITY_SHEET:
        sal_series = load_single_series_from_excel(SALINITY_PATH, sheet_name=SALINITY_SHEET, label="Salinity")
    else:
        sal_series = load_single_series_from_excel(SALINITY_PATH, sheet_name=None, label="Salinity")

    # Load tide 3 stations with robust fuzzy sheet matching
    tide_series = load_tide_three_stations(TIDE_PATH)

    # Align and filter to 2022 with common timestamps
    sal_aligned, tides_aligned = align_common_2022(sal_series, tide_series)

    # Compute correlations
    corrs = compute_correlations(sal_aligned, tides_aligned)

    # Make one figure
    make_figure(sal_aligned, tides_aligned, corrs, OUT_FIG_PATH)

    # Console summary
    print("[INFO] Aligned points (2022):", len(sal_aligned))
    for k in sorted(tides_aligned.keys()):
        print(f"[INFO] Corr(Salinity, {k}) = {corrs[k]}")

if __name__ == "__main__":
    main()
