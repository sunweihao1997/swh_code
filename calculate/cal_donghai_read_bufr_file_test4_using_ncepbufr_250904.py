#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import ncepbufr

BUFR_FILE = "/home/sun/data/satellite/satwind/20190413.satwnd/gdas.satwnd.t00z.20190413.bufr"   # ← 改成你的 BUFR 文件
OUT_NC    = "amv_points.nc"   # 输出 NetCDF 文件名

# 小工具
def read_arr(bufr, m):
    try:
        a = bufr.read_subset(m)
        if isinstance(a, np.ma.MaskedArray):
            a = a.filled(np.nan)
        return np.array(a, dtype=float, copy=False)
    except Exception:
        return np.array([])

def first_available(bufr, names):
    for n in names:
        arr = read_arr(bufr, n)
        if arr.size and np.isfinite(np.nanmax(arr)):
            return n, arr
    return None, np.array([])

def to_uv(wdir_deg, wspd):
    rad = np.deg2rad(wdir_deg)
    u = -wspd * np.sin(rad)
    v = -wspd * np.cos(rad)
    return u, v

# 候选经纬度/时间
LAT_CANDS = ["CLAT", "CLATH", "YOB", "LAT"]
LON_CANDS = ["CLON", "CLONH", "XOB", "LON"]

times, lats, lons, wdir, wspd, u_s, v_s = [], [], [], [], [], [], []

bufr = ncepbufr.open(BUFR_FILE)

while bufr.advance() == 0:
    # 消息参考时间（YYYYMMDDHH）
    ref_time = datetime.strptime(str(bufr.msg_date), "%Y%m%d%H")

    while bufr.load_subset() == 0:
        # 经纬度：在候选里找
        lat_name, LAT = first_available(bufr, LAT_CANDS)
        lon_name, LON = first_available(bufr, LON_CANDS)
        if LAT.size == 0 or LON.size == 0:
            continue

        # 时间：优先 DHR（相对小时）；否则 YEAR/MNTH/DAYS/HOUR/MINU/SECO；再不行用 RCYR/RCMO/…
        t = None
        DHR = read_arr(bufr, "DHR")
        if DHR.size and np.isfinite(DHR.flat[0]):
            t = ref_time + timedelta(seconds=float(DHR.flat[0]) * 3600.0)
        else:
            YEAR = read_arr(bufr, "YEAR"); MNTH = read_arr(bufr, "MNTH"); DAYS = read_arr(bufr, "DAYS")
            HOUR = read_arr(bufr, "HOUR"); MINU = read_arr(bufr, "MINU"); SECO = read_arr(bufr, "SECO")
            if all(x.size and np.isfinite(x.flat[0]) for x in [YEAR, MNTH, DAYS, HOUR, MINU]):
                ss = int(SECO.flat[0]) if SECO.size and np.isfinite(SECO.flat[0]) else 0
                try:
                    t = datetime(int(YEAR.flat[0]), int(MNTH.flat[0]), int(DAYS.flat[0]),
                                 int(HOUR.flat[0]), int(MINU.flat[0]), ss)
                except Exception:
                    t = None
            if t is None:
                RCYR = read_arr(bufr, "RCYR"); RCMO = read_arr(bufr, "RCMO"); RCDY = read_arr(bufr, "RCDY")
                RCHR = read_arr(bufr, "RCHR"); RCMN = read_arr(bufr, "RCMN")
                if all(x.size and np.isfinite(x.flat[0]) for x in [RCYR, RCMO, RCDY, RCHR, RCMN]):
                    t = datetime(int(RCYR.flat[0]), int(RCMO.flat[0]), int(RCDY.flat[0]),
                                 int(RCHR.flat[0]), int(RCMN.flat[0]), 0)
        if t is None:
            t = ref_time  # 兜底

        # 风向/风速
        WDIR = read_arr(bufr, "WDIR")
        WSPD = read_arr(bufr, "WSPD")
        if WDIR.size == 0 or WSPD.size == 0:
            continue

        # 取第一层（大多数 AMV 每条 subset 就一层；若有多层可改为循环）
        lat = float(LAT.flat[0]); lon = float(LON.flat[0])
        d   = float(WDIR.flat[0]); s   = float(WSPD.flat[0])
        if not (np.isfinite(lat) and np.isfinite(lon) and np.isfinite(d) and np.isfinite(s)):
            continue

        u, v = to_uv(d, s)

        times.append(np.datetime64(t))
        lats.append(lat); lons.append(lon)
        wdir.append(d);  wspd.append(s)
        u_s.append(u);   v_s.append(v)

bufr.close()

# 组织成 xarray Dataset（单维度：obs）
obs = np.arange(len(times), dtype="int64")
ds = xr.Dataset(
    data_vars=dict(
        lat=("obs", np.array(lats, dtype="float32")),
        lon=("obs", np.array(lons, dtype="float32")),
        wdir=("obs", np.array(wdir, dtype="float32")),
        wspd=("obs", np.array(wspd, dtype="float32")),
        u=("obs", np.array(u_s, dtype="float32")),
        v=("obs", np.array(v_s, dtype="float32")),
    ),
    coords=dict(
        obs=("obs", obs),
        time=("obs", np.array(times, dtype="datetime64[ns]")),
    ),
    attrs=dict(
        title="Atmospheric Motion Vectors decoded from BUFR",
        source="BUFR via NCEPLIBS-bufr (ncepbufr)",
        history="created by bufr_to_netcdf.py",
        Conventions="CF-1.8",
    ),
)

# 推荐的 CF 属性
ds["lat"].attrs.update(dict(standard_name="latitude", long_name="latitude", units="degrees_north"))
ds["lon"].attrs.update(dict(standard_name="longitude", long_name="longitude", units="degrees_east"))
ds["time"].attrs.update(dict(standard_name="time"))
ds["wdir"].attrs.update(dict(long_name="wind direction", units="degree"))
ds["wspd"].attrs.update(dict(long_name="wind speed", units="m s-1"))
ds["u"].attrs.update(dict(standard_name="eastward_wind", long_name="zonal wind component", units="m s-1"))
ds["v"].attrs.update(dict(standard_name="northward_wind", long_name="meridional wind component", units="m s-1"))

# 压缩/类型设置（可选）
encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} 
            for name in ["lat","lon","wdir","wspd","u","v"]}
encoding["time"] = {"zlib": True, "complevel": 4}

ds.to_netcdf(OUT_NC, format="NETCDF4", encoding=encoding)
print(f"OK -> {OUT_NC}, obs={ds.dims['obs']}")
