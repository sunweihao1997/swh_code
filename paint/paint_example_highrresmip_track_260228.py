'''
260228
This script is to plot examplr track simulated by highresmip model, to check the lat/lon data and the plotting.
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature


def to_180(lon):
    """Convert lon to [-180, 180)."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180) % 360) - 180


def main():
    parser = argparse.ArgumentParser(
        description="Read lat/lon from NetCDF and plot paired points over NW Pacific."
    )
    parser.add_argument("nc", help="Path to NetCDF file")
    parser.add_argument("--lat-var", default="lat", help="Latitude variable name (default: lat)")
    parser.add_argument("--lon-var", default="lon", help="Longitude variable name (default: lon)")

    # 默认给一个常用的西北太平洋范围，可自行改
    parser.add_argument("--lon-min", type=float, default=100.0, help="Min lon for plot/filter")
    parser.add_argument("--lon-max", type=float, default=180.0, help="Max lon for plot/filter")
    parser.add_argument("--lat-min", type=float, default=0.0, help="Min lat for plot/filter")
    parser.add_argument("--lat-max", type=float, default=60.0, help="Max lat for plot/filter")

    parser.add_argument("--lon-mode", choices=["auto", "0_360", "neg180_180"],
                        default="auto",
                        help="How to interpret/convert lon (default: auto).")
    parser.add_argument("--as-line", action="store_true",
                        help="Plot as a connected line (useful for tracks). Default: scatter points.")
    parser.add_argument("--title", default=None, help="Figure title")
    parser.add_argument("--save", default=None, help="Save figure to file (e.g., out.png). If not set, show.")
    parser.add_argument("--size", type=float, default=10, help="Marker size for scatter (default: 10)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Marker alpha (default: 0.7)")
    args = parser.parse_args()

    ds = xr.open_dataset(args.nc)

    if args.lat_var not in ds.variables or args.lon_var not in ds.variables:
        raise KeyError(f"Cannot find variables '{args.lat_var}'/'{args.lon_var}' in file. "
                       f"Available: {list(ds.variables)}")

    lat = ds[args.lat_var].values
    lon = ds[args.lon_var].values

    # squeeze + broadcast（防止维度多、但可配对的情况）
    lat = np.asarray(lat).squeeze()
    lon = np.asarray(lon).squeeze()
    lat, lon = np.broadcast_arrays(lat, lon)

    # 展平为点集合
    lat1 = lat.reshape(-1).astype(float)
    lon1 = lon.reshape(-1).astype(float)

    # 去 NaN
    ok = np.isfinite(lat1) & np.isfinite(lon1)
    lat1, lon1 = lat1[ok], lon1[ok]

    # lon 处理：auto 会根据数值范围判断
    if args.lon_mode == "auto":
        # 如果出现 >180 的值，基本可判断是 0–360
        lon_is_0360 = np.nanmax(lon1) > 180.0
        lon_mode = "0_360" if lon_is_0360 else "neg180_180"
    else:
        lon_mode = args.lon_mode

    if lon_mode == "neg180_180":
        lon_plot = to_180(lon1)
        # 过滤范围也需要在 [-180,180) 语系下
        lon_min = to_180(args.lon_min)
        lon_max = to_180(args.lon_max)

        # 处理跨越日期变更线的情况（如 lon_min=170, lon_max=-170）
        if lon_min <= lon_max:
            in_lon = (lon_plot >= lon_min) & (lon_plot <= lon_max)
        else:
            in_lon = (lon_plot >= lon_min) | (lon_plot <= lon_max)
    else:
        # 0–360 体系：把负经度映射到 0–360
        lon_plot = lon1 % 360
        lon_min = args.lon_min % 360
        lon_max = args.lon_max % 360
        if lon_min <= lon_max:
            in_lon = (lon_plot >= lon_min) & (lon_plot <= lon_max)
        else:
            in_lon = (lon_plot >= lon_min) | (lon_plot <= lon_max)

    in_lat = (lat1 >= args.lat_min) & (lat1 <= args.lat_max)
    mask = in_lon & in_lat

    lat_f = lat1[mask]
    lon_f = lon_plot[mask]

    if lat_f.size == 0:
        raise ValueError("No points left after filtering. Please check lon/lat range or lon-mode.")

    # --- Plot ---
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 6), dpi=150)
    ax = plt.axes(projection=proj)

    # set_extent 要和 lon_mode 一致
    ax.set_extent([lon_min, lon_max, args.lat_min, args.lat_max], crs=proj)

    ax.add_feature(cfeature.LAND, linewidth=0)
    ax.add_feature(cfeature.OCEAN, linewidth=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.6, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False

    if args.as_line:
        ax.plot(lon_f, lat_f, transform=proj, linewidth=1.2)
        ax.scatter(lon_f, lat_f, transform=proj, s=args.size * 0.1, alpha=args.alpha)
    else:
        ax.scatter(lon_f, lat_f, transform=proj, s=args.size*0.1, alpha=args.alpha)

    title = args.title or f"Paired (lat, lon) points in NW Pacific (N={lat_f.size})"
    ax.set_title(title)

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, bbox_inches="tight")
        print(f"Saved: {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()