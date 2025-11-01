#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
计算 ERA5 月尺度降水的 2022 年逐月距平 = 2022 每月 - 1980–2025 同月平均
要求目录下存在类似：era5-monthly_total_precipitation_YYYY.nc
"""

from pathlib import Path
import xarray as xr
import numpy as np
import re

# === 配置：修改为你的数据目录 ===
DATA_DIR = Path("/home/sun/wd_14/download_reanalysis/ERA5/monthly_single_0.5_0.5/downloads/era5-monthly")  # 改成你的文件夹路径
OUT_FILE = Path("/home/sun/data_n100/research/era5_precip_anomaly_2022.nc")  # 输出文件

# === 收集文件 ===
files = sorted(
    str(p) for p in DATA_DIR.glob("era5-monthly_total_precipitation_*.nc")
)
if not files:
    raise FileNotFoundError("未找到匹配的降水文件：era5-monthly_total_precipitation_*.nc")

# 按年份筛一遍，确保范围覆盖 1980–2025（容错：目录里多/少也能跑）
def _year_from_name(fp: str) -> int:
    m = re.search(r"(\d{4})\.nc$", fp)
    return int(m.group(1)) if m else -1

files = [f for f in files if 1980 <= _year_from_name(f) <= 2025]
if not files:
    raise RuntimeError("匹配到的文件不在 1980–2025 年范围内。")

print(f"将读取 {len(files)} 个文件，范围 {min(map(_year_from_name, files))}-{max(map(_year_from_name, files))}")

# === 读取数据（懒加载，自动启用 dask） ===
ds = xr.open_mfdataset(
    files,
    combine="by_coords",
    parallel=False,
    chunks=None,
    decode_times=True,
)

#print(ds)

## === 猜测降水变量名（优先 tp，否则选含 precip 的变量，否则取第一个数据变量） ===
data_vars = list(ds.data_vars)
if "tp" in data_vars:
    var = "tp"
else:
    candidates = [v for v in data_vars if "precip" in v.lower()]
    var = candidates[0] if candidates else data_vars[0]

print(f"使用变量：{var}")
#
## 保证时间坐标存在
if "valid_time" not in ds.dims and "valid_time" not in ds.coords:
    raise KeyError("数据中未找到 time 维度/坐标。")

# === 计算 1980–2025 的逐月气候态（同月平均） ===
clim_base = ds[var].sel(valid_time=slice("1980-01-01", "2025-12-31"))
# 注意：这些文件已是“月累计/总量”，这里对同一月份做多年平均即可
clim_monthly = clim_base.groupby("valid_time.month").mean("valid_time", skipna=True)

#print(clim_base)

## === 取 2022 年并计算距平 ===
p2022 = ds[var].sel(valid_time=slice("2022-01-01", "2022-12-31"))
## 广播按月份对齐：每个 2022-XX 减去对应 month 的 climatology
anom2022 = p2022.groupby("valid_time.month") - clim_monthly
anom2022 = anom2022.rename(f"{var}_anomaly")

## 可选：转换为 float32 以节省空间（保持 _FillValue/NaN）
anom2022 = anom2022.astype(np.float32)

# === 打包输出 ===
out_ds = anom2022.to_dataset()
# 传递部分元数据
out_ds[f"{var}_anomaly"].attrs.update({
    "long_name": f"{ds[var].attrs.get('long_name', var)} anomaly (relative to 1980–2025 monthly mean)",
    "units": ds[var].attrs.get("units", ""),  # 距平单位与原变量相同
    "comments": "Anomaly computed as 2022 monthly value minus 1980–2025 climatological monthly mean."
})
out_ds.attrs.update({
    "source": "ERA5 monthly",
    "method": "xarray groupby month, mean over 1980–2025, subtract for 2022",
})

# 按需设置压缩
encoding = {f"{var}_anomaly": {"zlib": True, "complevel": 4, "dtype": "float32"}}

out_ds.to_netcdf(OUT_FILE, encoding=encoding)
print(f"已写出距平文件：{OUT_FILE.resolve()}")
