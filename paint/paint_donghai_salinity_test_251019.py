#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
直接从指定路径绘制ERA5 runoff（ro）变量在中国地区的二维图。
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 文件路径和变量名
file_path = "/home/sun/data/runoff_test.nc"
var_name = "sro"

# 中国范围（经纬度）
CHINA_EXTENT = (73, 135, 18, 54)  # lon_min, lon_max, lat_min, lat_max

def to_lon180(ds):
    """把经度从(0,360)转换为(-180,180)，若已是 -180~180 则原样返回。"""
    lon_name = "lon" if "lon" in ds.coords else ("longitude" if "longitude" in ds.coords else None)
    if lon_name is None:
        raise ValueError("未找到经度坐标（lon 或 longitude）")
    lon = ds[lon_name]
    if lon.max() > 180:
        lon_new = ((lon + 180) % 360) - 180
        ds = ds.assign_coords({lon_name: lon_new}).sortby(lon_name)
    return ds, lon_name

def china_subset(da, lon_name, lat_name):
    lon_min, lon_max, lat_min, lat_max = CHINA_EXTENT
    # 处理越界切片（已是 -180~180）
    da = da.sel({lon_name: slice(lon_min, lon_max), lat_name: slice(lat_max, lat_min)}) if da[lat_name][0] > da[lat_name][-1] \
        else da.sel({lon_name: slice(lon_min, lon_max), lat_name: slice(lat_min, lat_max)})
    return da

# 打开 NetCDF 文件
ds = xr.open_dataset(file_path)

# 查找变量
if var_name in ds:
    da = ds[var_name]
else:
    raise ValueError(f"文件中未找到指定变量 '{var_name}'")

# 标准化经纬度命名
ds, lon_name = to_lon180(ds)
lat_name = "lat" if "lat" in ds.coords else ("latitude" if "latitude" in ds.coords else None)
if lat_name is None:
    raise ValueError("未找到纬度坐标（lat 或 latitude）")

# 截取中国区域数据
da_cn = china_subset(da, lon_name, lat_name)

# 单位转换：将 m 转为 mm（如果单位是 m）
units = da_cn.attrs.get("units", "")
if units == "m":
    da_cn = da_cn * 1000.0
    units = "mm"

# 计算颜色范围（robust）
vmin = float(da_cn.quantile(0.02).values)
vmax = float(da_cn.quantile(0.98).values)
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin, vmax = None, None

# 作图
proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(8.6, 6.8), dpi=150)
ax = plt.axes(projection=proj)
ax.set_extent(CHINA_EXTENT, crs=proj)

# 地理要素
ax.add_feature(cfeature.LAND, alpha=0.1)
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.LAKES, edgecolor="black", facecolor="none", linewidth=0.3)
gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--")
gl.top_labels = False
gl.right_labels = False

# 等值面填色
im = ax.pcolormesh(
    da_cn[lon_name], da_cn[lat_name], da_cn[0],
    transform=proj, shading="auto", vmin=vmin, vmax=vmax,
)
cb = plt.colorbar(im, ax=ax, fraction=0.032, pad=0.04)
cb.set_label(f"Runoff ({units})")

# 标题
vlabel = var_name.upper()
plt.title(f"ERA5 Runoff ({vlabel}) over China", fontsize=10)

# 保存图片
out_png = file_path.replace(".nc", "_runoff_china.png")
plt.tight_layout()
fig.savefig(out_png, bbox_inches="tight")
print(f"图像已保存为: {out_png}")
