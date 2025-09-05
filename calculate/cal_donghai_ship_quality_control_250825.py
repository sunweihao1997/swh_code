'''
2025-8-25
calculate QC-index for Donghai ship data

Task Link: https://chatgpt.com/c/68a7dc8c-7860-8331-945c-e72eaf9b1e61
'''
import numpy as np
import pandas as pd
import xarray as xr
import os
import chardet
from datetime import datetime
import pytz
import re

path_data = "/mnt/f/ERA5_ship/donghai_ship_true_wind/"
list_file = os.listdir(path_data)

# QC flags
GOOD, SUSPECT, FAIL, MISSING = 1, 3, 4, 9

def _flag_series(init_index):
    return pd.Series(GOOD, index=init_index, dtype="int64")

def _combine_overall(row_vals):
    """
    总体QC: 如果有 FAIL(4) 就返回 FAIL;
           否则如果有 SUSPECT(3) 就返回 SUSPECT;
           否则如果有 MISSING(9) 就返回 MISSING;
           否则 GOOD。
    """
    vals = set(row_vals.values.astype(int))
    if FAIL in vals:
        return FAIL
    elif SUSPECT in vals:
        return SUSPECT
    elif MISSING in vals:
        return MISSING
    else:
        return GOOD

def qc_from_table(df,
                  col_p="气压",      # hPa
                  col_rh="湿度",     # %
                  col_t="气温",      # K 或 ℃
                  col_ws="风速",     # m/s
                  col_wd="风向"      # degree (0–360)
                 ):
    out = df.copy()

    # ---------- 气压 ----------
    if col_p in out.columns:
        p = out[col_p]
        f = _flag_series(out.index)
        f[p.isna()] = MISSING
        f[(p < 920) | (p > 1085)] = FAIL
        out["qc_气压"] = f

    # ---------- 相对湿度 ----------
    if col_rh in out.columns:
        rh = out[col_rh]
        f = _flag_series(out.index)
        f[rh.isna()] = MISSING
        f[(rh < 10) | (rh > 105)] = FAIL
        f[(rh > 100) & (rh <= 105)] = SUSPECT
        out["qc_湿度"] = f

    # ---------- 气温 ----------
    if col_t in out.columns:
        T = out[col_t].astype(float)
        Tc = T - 273.15 if np.nanmedian(T) > 200 else T
        f = _flag_series(out.index)
        f[T.isna()] = MISSING
        f[(Tc < -50) | (Tc > 50)] = FAIL
        f[(Tc > 40) & (Tc <= 50)] = SUSPECT
        out["qc_气温"] = f

    # ---------- 风速 ----------
    if col_ws in out.columns:
        ws = out[col_ws]
        f = _flag_series(out.index)
        f[ws.isna()] = MISSING
        f[(ws < 0) | (ws > 75)] = FAIL
        f[(ws > 60) & (ws <= 75)] = SUSPECT
        out["qc_风速"] = f

    # ---------- 风向 ----------
    if col_wd in out.columns:
        wd = out[col_wd]
        f = _flag_series(out.index)
        f[wd.isna()] = MISSING
        f[(wd < 0) | (wd > 360)] = FAIL
        out["qc_风向"] = f

    # ---------- 总体标志 ----------
    qc_cols = [c for c in out.columns if c.startswith("qc_")]
    if qc_cols:
        out["qc_overall"] = out[qc_cols].apply(_combine_overall, axis=1)

    return out, qc_cols

# exclude other data
for filename in list_file:
    if not filename.endswith('.csv'):
        list_file.remove(filename)

for file in list_file:
    print(f"Processing file: {file}")
    df = pd.read_csv(path_data + file, encoding='utf-8-sig')

    qc_df, qc_cols = qc_from_table(df)

    #print(qc_df)

    qc_df.to_csv("/mnt/f/ERA5_ship/donghai_ship_ERA5_QC/" + file, index=False, encoding='utf-8-sig')
    #print(f"Finished processing and saved: {file}")