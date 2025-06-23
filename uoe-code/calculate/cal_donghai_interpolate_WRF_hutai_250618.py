'''
2025-6-18
This script is to interpolate WRF output onto hutai location
'''
#from netCDF4 import Dataset
import numpy as np
from wrf import getvar, interplevel, interpline, interplevel, latlon_coords, ll_to_xy
import xarray as xr

# 打开WRF输出文件
#wrf_path = "/mnt/Dell_NAS/WRFOUT/"
#
#
#ds = xr.open_dataset(wrf_path + "2021/" + "wrfout_d02_2021-08-05_00", engine="scipy")
#t2 = getvar(ds, "T2")
#
#
## 选择你想插值的变量，比如温度 T2（2米温度）
#t2 = getvar(wrf_file, "T2")
#
## 获取经纬度
#lats, lons = latlon_coords(t2)
#
## 要插值的经纬度坐标
#target_lat = 34.5
#target_lon = 112.3
#
## 将经纬度转换为最近的网格索引（不是插值）
## xy 返回的是 [x, y]（注意顺序）
#xy = ll_to_xy(wrf_file, target_lat, target_lon)
#ix, iy = int(xy[0]), int(xy[1])
#
## 获取最近网格点的值（非插值）
#nearest_value = t2[iy, ix]
#
#print(f"Nearest T2 value at ({target_lat}, {target_lon}): {nearest_value.values} K")
#
## ⚠️ 注意：如果你想要 **真正插值（而不是取最近网格点）**，请看下一步
#