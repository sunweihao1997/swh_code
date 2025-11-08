#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# === 配置：你的 NetCDF 路径 ===
nc_path = "/home/sun/data/bufr_test/message_1000.nc"

# === 读取数据 ===
ds   = xr.load_dataset(nc_path)
lat  = ds["CLAT"].values.astype(float)
lon  = ds["CLON"].values.astype(float)
wdir = ds["WDIR"].values.astype(float)   # 风向（来自方向，度，气象角：北=0，顺时针）
wspd = ds["WSPD"].values.astype(float)   # 风速（m/s 或原单位，根据BUFR）

# 去掉缺测
mask = ~(np.isfinite(lat) & np.isfinite(lon) & np.isfinite(wdir) & np.isfinite(wspd))
lat, lon, wdir, wspd = lat[~mask], lon[~mask], wdir[~mask], wspd[~mask]

# === 风向风速 -> U/V（气象约定：来自方向）===
# 方向单位：度；0度=北风；顺时针增大
rad  = np.deg2rad(wdir)
u = -wspd * np.sin(rad)   # 向东分量
v = -wspd * np.cos(rad)   # 向北分量

# === 简单抽稀，避免太密 ===
step = max(1, int(len(lat) / 1500))   # 目标最多 ~1500 箭头，可调
lat, lon, u, v, wspd = lat[::step], lon[::step], u[::step], v[::step], wspd[::step]

# === 优先尝试 cartopy 作底图 ===
used_cartopy = False
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    used_cartopy = True
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=proj)
    ax.set_global()
    ax.coastlines(linewidth=0.6)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)

    q = ax.quiver(
        lon, lat, u, v,
        wspd,  # 颜色映射用风速
        transform=ccrs.PlateCarree(),
        scale=700,     # 缩放（可调）
        width=0.0025,  # 箭杆宽度（可调）
        alpha=0.9
    )
    cb = plt.colorbar(q, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
    cb.set_label("Wind speed")

    ax.set_title("BUFR AMV/Surface Winds — message vectors", fontsize=12)
    plt.tight_layout()

except Exception:
    # === 退化：普通 matplotlib 平面图 ===
    fig, ax = plt.subplots(figsize=(10, 6))
    q = ax.quiver(
        lon, lat, u, v,
        wspd,
        scale=700,
        width=0.0025,
        alpha=0.9
    )
    cb = plt.colorbar(q, ax=ax, orientation="vertical", pad=0.02)
    cb.set_label("Wind speed")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_title("BUFR AMV/Surface Winds — message vectors (plain lat/lon)")

plt.show()
