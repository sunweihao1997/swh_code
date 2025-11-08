#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
绘制中国地区 2022 年 6–9 月降水异常图（4 张子图）
输入：era5_precip_anomaly_2022.nc
输出：era5_precip_anomaly_2022_JJAS.png
"""

import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from pathlib import Path

# === 输入文件路径 ===
IN_FILE = Path("/home/sun/data_n100/research/era5_precip_anomaly_2022.nc")

# === 读取数据 ===
ds = xr.open_dataset(IN_FILE)
# 猜测距平变量名（一般为 *_anomaly）
var_name = [v for v in ds.data_vars if "anom" in v.lower() or "anomaly" in v.lower()][0]
da = ds[var_name]

# === 提取中国区域（大致范围） ===
da_china = da.sel(longitude=slice(70, 140), latitude=slice(55, 15))  # 注意ERA5纬度通常是递减的

# === 选取月份 ===
months = [5, 6, 7, 8, 9, 10]
da_sel = da_china.sel(valid_time=da_china["valid_time"].dt.month.isin(months))

# === 画图 ===
fig, axes = plt.subplots(
    2, 3,
    figsize=(12, 10),
    subplot_kw={'projection': ccrs.PlateCarree()}
)
fig.suptitle("ERA5 2022 6-9 Precipitation Anomaly", fontsize=16, y=0.93)

for i, month in enumerate(months):
    ax = axes.flat[i]
    data = da_sel.sel(valid_time=str(f"2022-{month:02d}")).squeeze() * 1000 *30
    im = data.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-150, vmax=150,  # 调整色标范围，单位m
        add_colorbar=False,
        add_labels=False,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAKES, edgecolor='gray', facecolor='none', linewidth=0.3)
    ax.add_feature(cfeature.RIVERS, edgecolor='gray', linewidth=0.3)
    ax.set_extent([70, 140, 15, 55], crs=ccrs.PlateCarree())
    ax.set_title(f"{month}", fontsize=13)

# 统一色标
cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
plt.colorbar(im, cax=cbar_ax, orientation="horizontal", label="Precipitation Anomaly (m)")

# === 保存输出 ===
out_png = "/home/sun/paint/donghai/era5_precip_anomaly_2022_JJAS.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.close()
#print(f"✅ 已保存图像: {out_png.resolve()}")
