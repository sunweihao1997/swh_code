#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ncepbufr
import numpy as np
import xarray as xr
from datetime import datetime

# === 配置部分 ===
bufr_file = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
msg_idx   = 1000
out_file  = "/home/sun/data/bufr_test/message_1000.nc"


def safe_read_scalar(b, mnem, default=np.nan):
    """读取单个助记词的首值；不存在或缺测则返回 default"""
    try:
        arr = b.read_subset(mnem)
        if arr is None:
            return default
        a = np.ravel(arr)
        if a.size == 0:
            return default
        v = a[0]
        if np.ma.is_masked(v):
            return default
        return float(v)
    except Exception:
        return default


def goto_message(b, msg_idx):
    """advance到指定message（1-based）"""
    cur = 0
    while b.advance() == 0:
        cur += 1
        if cur == msg_idx:
            return True
    return False


# === 主流程 ===
b = ncepbufr.open(bufr_file)
if not goto_message(b, msg_idx):
    b.close()
    raise SystemExit(f"Message {msg_idx} not found in {bufr_file}")

times, lats, lons, wdirs, wspds = [], [], [], [], []

while b.load_subset() == 0:
    year  = safe_read_scalar(b, "YEAR", np.nan)
    month = safe_read_scalar(b, "MNTH", np.nan)
    day   = safe_read_scalar(b, "DAYS", np.nan)
    hour  = safe_read_scalar(b, "HOUR", np.nan)
    minu  = safe_read_scalar(b, "MINU", 0.0)
    seco  = safe_read_scalar(b, "SECO", 0.0)

    if any(np.isnan([year, month, day, hour])):
        continue
    try:
        t = datetime(int(year), int(month), int(day), int(hour), int(minu), int(seco))
    except Exception:
        continue

    clat = safe_read_scalar(b, "CLAT", np.nan)
    clon = safe_read_scalar(b, "CLON", np.nan)
    wdir = safe_read_scalar(b, "WDIR", np.nan)
    wspd = safe_read_scalar(b, "WSPD", np.nan)

    times.append(t)
    lats.append(clat)
    lons.append(clon)
    wdirs.append(wdir)
    wspds.append(wspd)

b.close()

n = len(times)
if n == 0:
    raise SystemExit("No valid subsets collected.")

ds = xr.Dataset(
    data_vars=dict(
        WDIR=("obs", np.array(wdirs, dtype="float32")),
        WSPD=("obs", np.array(wspds, dtype="float32")),
        CLAT=("obs", np.array(lats,  dtype="float32")),
        CLON=("obs", np.array(lons,  dtype="float32")),
    ),
    coords=dict(
        obs=np.arange(n, dtype="int32"),
        time=("obs", np.array(times, dtype="datetime64[ns]")),
    ),
    attrs=dict(
        title="BUFR message to NetCDF",
        source_file=bufr_file,
        message_index=str(msg_idx),
        note="MINU/SECO default to 0 if absent; missing values are NaN.",
    )
)

#ds.to_netcdf(out_file)
#print(f"[OK] 已写出 {out_file}，包含 {n} 条观测")
#
#print(ds['time'].data)
#print(ds['WDIR'].data)
#print(ds['CLAT'].data)
#print(ds['CLON'].data)

# ========== Another Test ==========
ds = xr.open_dataset("/home/sun/data/bufr_test/message_00243_NC005010.nc")
print(ds)
print(ds['msg_type'].data)