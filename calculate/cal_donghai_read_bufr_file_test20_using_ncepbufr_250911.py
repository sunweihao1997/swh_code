#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# ======== 目录与过滤规则（按需改动）========
IN_DIR = "/home/sun/data/bufr_merged_1hourwindow"
FILE_PATTERN = "*_gridded.nc"   # 只处理“格点化”结果
BBOX = None                     # (lon_min, lon_max, lat_min, lat_max) 或 None 不裁剪
TARGET_ARROWS = 1800            # 自动抽稀的目标箭头数
QUIVER_SCALE = 700              # 箭头缩放
# =========================================

def autoset_thin(nlat, nlon, target_arrows=1800):
    if nlat == 0 or nlon == 0:
        return 1
    step = int(np.sqrt((nlat * nlon) / max(1, target_arrows)))
    return max(1, step)

def ensure_2d_latlon(da, lat_name="lat", lon_name="lon"):
    """
    把 DataArray 调整成 (lat, lon) 顺序的二维数组。
    若不是二维或缺坐标，返回 None。
    """
    if not hasattr(da, "dims"):
        return None
    dims = list(da.dims)
    if lat_name in dims and lon_name in dims and len(dims) == 2:
        # 按 (lat, lon) 排序
        if tuple(dims) != (lat_name, lon_name):
            return da.transpose(lat_name, lon_name)
        return da
    return None

def met_to_uv(wdir_deg, wspd):
    th = np.deg2rad(wdir_deg)
    u = -wspd * np.sin(th)
    v = -wspd * np.cos(th)
    return u, v

def save_quiver(lon, lat, U2, V2, title, out_png, bbox=None, thin_every=None):
    assert U2.ndim == 2 and V2.ndim == 2
    if thin_every is None:
        thin_every = autoset_thin(U2.shape[0], U2.shape[1], TARGET_ARROWS)

    lon_q = lon[::thin_every]
    lat_q = lat[::thin_every]
    U_q   = U2[::thin_every, ::thin_every]
    V_q   = V2[::thin_every, ::thin_every]

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        proj = ccrs.PlateCarree()
        fig = plt.figure(figsize=(10, 6))
        ax = plt.axes(projection=proj)
        ax.coastlines(linewidth=0.6)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.4)
        ax.quiver(lon_q, lat_q, U_q, V_q, transform=ccrs.PlateCarree(),
                  scale=QUIVER_SCALE, width=0.0025, alpha=0.9)
        if bbox is not None:
            ax.set_extent([bbox[0], bbox[1], bbox[2], bbox[3]], crs=ccrs.PlateCarree())
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.quiver(lon_q, lat_q, U_q, V_q, scale=QUIVER_SCALE, width=0.0025, alpha=0.9)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(True, ls="--", alpha=0.3)
        if bbox is not None:
            ax.set_xlim(bbox[0], bbox[1]); ax.set_ylim(bbox[2], bbox[3])
        ax.set_title(title, fontsize=11)
        plt.tight_layout()
        fig.savefig(out_png, dpi=150)
        plt.close(fig)

def main():
    files = sorted(glob.glob(os.path.join(IN_DIR, FILE_PATTERN)))
    if not files:
        print(f"未找到文件：{os.path.join(IN_DIR, FILE_PATTERN)}")
        return

    for f in files:
        try:
            ds = xr.load_dataset(f)
        except Exception as e:
            print(f"[跳过] 读取失败 {f}: {e}")
            continue

        # 经纬度坐标名
        if "lat" in ds and "lon" in ds:
            lat_name, lon_name = "lat", "lon"
        elif "CLAT" in ds and "CLON" in ds:
            lat_name, lon_name = "CLAT", "CLON"
        else:
            print(f"[跳过] {f} 未找到经纬度坐标（lat/lon 或 CLAT/CLON）")
            continue

        lat = ds[lat_name].values
        lon = ds[lon_name].values

        # 优先使用 U/V；若没有则由 2D 的 WDIR/WSPD 现算
        has_uv = ("U" in ds) and ("V" in ds)
        has_dirspd = ("WDIR" in ds) and ("WSPD" in ds)

        base = os.path.splitext(os.path.basename(f))[0]

        if "time" in ds.dims:
            nt = ds.sizes["time"]
            for ti in range(nt):
                # 取出某时次
                if has_uv:
                    U_da = ensure_2d_latlon(ds["U"].isel(time=ti), lat_name, lon_name)
                    V_da = ensure_2d_latlon(ds["V"].isel(time=ti), lat_name, lon_name)
                elif has_dirspd:
                    WDIR_da = ensure_2d_latlon(ds["WDIR"].isel(time=ti), lat_name, lon_name)
                    WSPD_da = ensure_2d_latlon(ds["WSPD"].isel(time=ti), lat_name, lon_name)
                    if (WDIR_da is not None) and (WSPD_da is not None):
                        U_vals, V_vals = met_to_uv(WDIR_da.values, WSPD_da.values)
                        U_da = xr.DataArray(U_vals, coords=WDIR_da.coords, dims=WDIR_da.dims)
                        V_da = xr.DataArray(V_vals, coords=WSPD_da.coords, dims=WSPD_da.dims)
                    else:
                        U_da = V_da = None
                else:
                    print(f"[跳过] {f} 缺 U/V 也缺 WDIR/WSPD。")
                    break

                if (U_da is None) or (V_da is None):
                    print(f"[跳过] {f} time[{ti}] 不是二维格点（或变量缺失）。")
                    continue

                # 裁剪（可选）
                if BBOX is not None:
                    lon_min, lon_max, lat_min, lat_max = BBOX
                    j_sel = np.where((lon >= lon_min) & (lon <= lon_max))[0]
                    i_sel = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                    if j_sel.size == 0 or i_sel.size == 0:
                        print(f"[提示] {f} time[{ti}] 裁剪后无有效网格，跳过。")
                        continue
                    lon_ = lon[j_sel]; lat_ = lat[i_sel]
                    U2 = U_da.values[np.ix_(i_sel, j_sel)]
                    V2 = V_da.values[np.ix_(i_sel, j_sel)]
                else:
                    lon_, lat_ = lon, lat
                    U2, V2 = U_da.values, V_da.values

                tstr_h = np.datetime_as_string(ds["time"].values[ti], unit='m')  # 可读
                tstr_fn = tstr_h.replace(':','').replace('T','_')                # 文件名安全
                png_path = os.path.join(IN_DIR, f"{base}_t{tstr_fn}.png")
                title = f"{base} | time={tstr_h}"
                save_quiver(lon_, lat_, U2, V2, title, png_path, bbox=BBOX)
                print(f"[OK] 保存 {png_path}")

        else:
            # 无 time 维：要求变量就是二维
            if has_uv:
                U_da = ensure_2d_latlon(ds["U"], lat_name, lon_name)
                V_da = ensure_2d_latlon(ds["V"], lat_name, lon_name)
            elif has_dirspd:
                WDIR_da = ensure_2d_latlon(ds["WDIR"], lat_name, lon_name)
                WSPD_da = ensure_2d_latlon(ds["WSPD"], lat_name, lon_name)
                if (WDIR_da is not None) and (WSPD_da is not None):
                    U_vals, V_vals = met_to_uv(WDIR_da.values, WSPD_da.values)
                    U_da = xr.DataArray(U_vals, coords=WDIR_da.coords, dims=WDIR_da.dims)
                    V_da = xr.DataArray(V_vals, coords=WSPD_da.coords, dims=WSPD_da.dims)
                else:
                    U_da = V_da = None
            else:
                print(f"[跳过] {f} 缺 U/V 也缺 WDIR/WSPD。")
                return

            if (U_da is None) or (V_da is None):
                print(f"[跳过] {f} 不是二维格点（或变量缺失）。")
                return

            if BBOX is not None:
                lon_min, lon_max, lat_min, lat_max = BBOX
                j_sel = np.where((lon >= lon_min) & (lon <= lon_max))[0]
                i_sel = np.where((lat >= lat_min) & (lat <= lat_max))[0]
                if j_sel.size == 0 or i_sel.size == 0:
                    print(f"[提示] {f} 裁剪后无有效网格，跳过。")
                    return
                lon_ = lon[j_sel]; lat_ = lat[i_sel]
                U2 = U_da.values[np.ix_(i_sel, j_sel)]
                V2 = V_da.values[np.ix_(i_sel, j_sel)]
            else:
                lon_, lat_ = lon, lat
                U2, V2 = U_da.values, V_da.values

            png_path = os.path.join(IN_DIR, f"{base}.png")
            save_quiver(lon_, lat_, U2, V2, base, png_path, bbox=BBOX)
            print(f"[OK] 保存 {png_path}")

if __name__ == "__main__":
    main()
