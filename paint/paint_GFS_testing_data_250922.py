# -*- coding: utf-8 -*-
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

# ========= 参数设置 =========
nc_dir = "/public/share/sunweihao/nc_out"   # 输入 nc 文件所在目录
out_dir = "/public/home/sunweihao/plot"     # 输出图片目录
file_list = [
    "gfs_3_20250421_0000_000.nc4", "gfs_3_20250422_0000_000.nc4", "gfs_3_20250423_0000_000.nc4",
    "gfs_3_20250424_0000_000.nc4", "gfs_3_20250425_0000_000.nc4",
    "gfs_3_20250421_0600_000.nc4", "gfs_3_20250422_0600_000.nc4", "gfs_3_20250423_0600_000.nc4",
    "gfs_3_20250424_0600_000.nc4", "gfs_3_20250425_0600_000.nc4",
    "gfs_3_20250421_1200_000.nc4", "gfs_3_20250422_1200_000.nc4", "gfs_3_20250423_1200_000.nc4",
    "gfs_3_20250424_1200_000.nc4", "gfs_3_20250425_1200_000.nc4",
    "gfs_3_20250421_1800_000.nc4", "gfs_3_20250422_1800_000.nc4", "gfs_3_20250423_1800_000.nc4",
    "gfs_3_20250424_1800_000.nc4", "gfs_3_20250425_1800_000.nc4"
]
target_level_pa = 85000
draw_region = [70, 140, 15, 55]  # 东亚区域
# ==========================

# 打开 NetCDF
def open_nc(path):
    try:
        return xr.open_dataset(path, engine="netcdf4")
    except Exception:
        try:
            return xr.open_dataset(path, engine="h5netcdf")
        except Exception:
            return xr.open_dataset(path, engine="scipy")

# 绘图函数
def plot_file(ncfile, outfile):
    ds = open_nc(ncfile)
    lat = ds["lat_0"]; lon = ds["lon_0"]

    # 风场
    u3d = ds["UGRD_P0_L100_GLL0"]
    v3d = ds["VGRD_P0_L100_GLL0"]
    u850 = u3d.sel(lv_ISBL0=target_level_pa, method="nearest")
    v850 = v3d.sel(lv_ISBL0=target_level_pa, method="nearest")
    level_used = float(u850["lv_ISBL0"])

    # 海平面气压 (Pa → hPa)
    mslp = ds["PRMSL_P0_L101_GLL0"] / 100.0

    # 拼接经度
    mslp_cyc, lon_cyc = add_cyclic_point(mslp.values, coord=lon.values)
    u850_cyc, _ = add_cyclic_point(u850.values, coord=lon.values)
    v850_cyc, _ = add_cyclic_point(v850.values, coord=lon.values)
    Lon2d, Lat2d = np.meshgrid(lon_cyc, lat.values)

    # 绘图
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(12, 7), dpi=150)
    ax = plt.axes(projection=proj)

    if draw_region is None:
        ax.set_global()
    else:
        ax.set_extent(draw_region, crs=proj)

    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, linestyle="--")
    gl.right_labels = False; gl.top_labels = False

    # 填色 MSLP，间隔 2.5 hPa
    vmin = np.floor(np.nanmin(mslp_cyc) / 2.5) * 2.5
    vmax = np.ceil(np.nanmax(mslp_cyc) / 2.5) * 2.5
    levels = np.arange(vmin, vmax + 0.01, 2.5)
    cf = ax.contourf(Lon2d, Lat2d, mslp_cyc, levels=levels,
                     cmap="Spectral_r", extend="both", transform=proj)
    cb = plt.colorbar(cf, ax=ax, orientation="horizontal", pad=0.04, aspect=50)
    cb.set_label("MSLP (hPa)")

    # 风矢量（更密更长）
    skip = (slice(None, None, 2), slice(None, None, 2))  # 每2格取一点
    q = ax.quiver(Lon2d[skip], Lat2d[skip], u850_cyc[skip], v850_cyc[skip],
                  transform=proj, scale=250, width=0.0025, pivot="middle")
    ax.quiverkey(q, 0.92, -0.02, 10, "10 m s$^{-1}$", labelpos="E", coordinates="axes")

    # 标题
    plt.title(f"GFS {os.path.basename(ncfile)}\n"
              f"MSLP (filled, 2.5 hPa) & 850hPa Wind | Level used: {level_used/100:.0f} hPa",
              fontsize=11)
    plt.tight_layout()
    plt.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"图已保存: {outfile}")

# 主程序
if __name__ == "__main__":
    os.makedirs(out_dir, exist_ok=True)
    for fname in file_list:
        infile = os.path.join(nc_dir, fname)
        outfile = os.path.join(out_dir, fname.replace(".nc4", ".png"))
        plot_file(infile, outfile)
