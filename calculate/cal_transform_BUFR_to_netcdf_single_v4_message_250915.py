#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from datetime import datetime
import numpy as np
import xarray as xr
import ncepbufr

# ===== 固定配置（测试用）=====
bufr_file = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"
out_dir   = "/home/sun/data/bufr_test"
os.makedirs(out_dir, exist_ok=True)

def safe_read_scalar(b, mnem, default=np.nan):
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

def process_current_message(b, imsg, msg_type_str):
    times, lats, lons, wdirs, wspds = [], [], [], [], []

    while True:
        rc = b.load_subset()
        if rc != 0:
            break

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
        if np.isnan(clat) and np.isnan(clon):
            clat = safe_read_scalar(b, "CLATH", np.nan)
            clon = safe_read_scalar(b, "CLONH", np.nan)

        wdir = safe_read_scalar(b, "WDIR", np.nan)
        wspd = safe_read_scalar(b, "WSPD", np.nan)

        times.append(t); lats.append(clat); lons.append(clon); wdirs.append(wdir); wspds.append(wspd)

    if len(times) == 0:
        return None

    ds = xr.Dataset(
        data_vars=dict(
            WDIR=("obs", np.array(wdirs, dtype="float32")),
            WSPD=("obs", np.array(wspds, dtype="float32")),
            CLAT=("obs", np.array(lats,  dtype="float32")),
            CLON=("obs", np.array(lons,  dtype="float32")),
            msg_type=([], np.array(msg_type_str, dtype="S")),  # ← 原样保存为字符串
        ),
        coords=dict(
            obs=np.arange(len(times), dtype="int32"),
            time=("obs", np.array(times, dtype="datetime64[ns]")),
        ),
        attrs=dict(
            title="BUFR message to NetCDF (one file per message)",
            source_file=bufr_file,
            message_index=str(imsg),
            msg_type=msg_type_str,  # ← 全局属性也写入
        )
    )
    return ds

def main():
    b = ncepbufr.open(bufr_file)
    imsg = 0
    saved = 0

    while b.advance() == 0:
        imsg += 1

        # 不转换为 int，直接保留字符串
        msg_type_str = str(b.msg_type)

        ds = process_current_message(b, imsg, msg_type_str)
        if ds is None:
            print(f"[跳过] Message {imsg:05d} —— 无有效观测")
            continue

        # 文件名也保留完整字符串
        out_path = os.path.join(out_dir, f"message_{imsg:05d}_{msg_type_str}.nc")
        ds.to_netcdf(out_path)
        saved += 1
        print(f"[OK] Message {imsg:05d} ({msg_type_str}) -> {out_path} (nobs={ds.dims['obs']})")

    b.close()
    print(f"\n完成：共导出 {saved} 个 NetCDF 文件，目录：{out_dir}")

if __name__ == "__main__":
    main()
