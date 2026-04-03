import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# 海岸线（需要 cartopy）
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

# =========================
# 1) 用户参数
# =========================
nc_path = "/home/sun/data/download_data/other/273f3492b051e89a1ebd10f836518a7d.nc"
var_name = "t2m"
out_pdf = "/home/sun/paint/donghai/typhoon_prediction/t2m_contourf.pdf"

convert_K_to_C = True
n_levels = 21

# 排除极地：只绘制该纬度范围（可按需改）
lat_min, lat_max = -60, 60

# =========================
# 2) 读数据并做基础检查
# =========================
ds = xr.open_dataset(nc_path)

if var_name not in ds:
    raise KeyError(f"变量 {var_name} 不在数据集中。可用变量：{list(ds.data_vars)}")

da = ds[var_name]

needed_dims = {"valid_time", "latitude", "longitude"}
if not needed_dims.issubset(set(da.dims)):
    raise ValueError(
        f"{var_name} 的维度是 {da.dims}，未包含 {needed_dims}。"
        "请检查维度命名是否不同（例如 time/lat/lon）。"
    )

lats = ds["latitude"]
lons = ds["longitude"].values
times = ds["valid_time"].values

# 转单位（可选）
plot_da = da
unit_label = "K"
if convert_K_to_C:
    plot_da = da - 273.15
    unit_label = "°C"

# =========================
# 3) 裁剪纬度范围（排除极地）
# =========================
if float(lats[0]) < float(lats[-1]):
    plot_da = plot_da.sel(latitude=slice(lat_min, lat_max))
else:
    plot_da = plot_da.sel(latitude=slice(lat_max, lat_min))

lats_sel = plot_da["latitude"].values

# =========================
# 3.1) 修正经度：统一到 [-180, 180) 并排序（避免 0°缝在图中央）
# =========================
lons_180 = ((lons + 180.0) % 360.0) - 180.0          # 0..360 -> -180..180
sort_idx = np.argsort(lons_180)
lons_sorted = lons_180[sort_idx]

# 统一色标（基于裁剪后的所有 valid_time；仍然使用原始经度次序也没问题）
vmin = float(np.nanmin(plot_da.values))
vmax = float(np.nanmax(plot_da.values))
levels = np.linspace(vmin, vmax, n_levels)

# =========================
# 4) 输出：每个 valid_time 一个 PDF（共 3 个）
# =========================
out_dir = os.path.dirname(out_pdf)
base_name = os.path.splitext(os.path.basename(out_pdf))[0]

# 地理范围（注意：经度使用 [-180, 180)）
lon_min_eff, lon_max_eff = float(np.min(lons_sorted)), float(np.max(lons_sorted))
lat_min_eff, lat_max_eff = float(np.min(lats_sel)), float(np.max(lats_sel))

for i, t in enumerate(times):
    # 取单个时次切片：(latitude, longitude)
    slice2d = plot_da.isel(valid_time=i).values

    # 关键：按经度排序重排数据列
    slice2d_sorted = slice2d[:, sort_idx]

    # 关键：经度方向加 cyclic point，消除拼接白线
    slice2d_cyc, lons_cyc = add_cyclic_point(slice2d_sorted, coord=lons_sorted)

    fig = plt.figure(figsize=(11, 4.5), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree())

    ax.set_extent([lon_min_eff, lon_max_eff, lat_min_eff, lat_max_eff],
                  crs=ccrs.PlateCarree())

    cf = ax.contourf(
        lons_cyc, lats_sel, slice2d_cyc,
        levels=levels, extend="both",
        cmap="coolwarm",
        transform=ccrs.PlateCarree()
    )

    # 海岸线
    ax.coastlines(resolution="110m", linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    cbar = fig.colorbar(cf, ax=ax, orientation="vertical", pad=0.02, shrink=0.95)
    cbar.set_label(f"{var_name} ({unit_label})")

    # 不要标题（按你的要求）
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    out_pdf_i = os.path.join(out_dir, f"{base_name}_t{i}.pdf")
    with PdfPages(out_pdf_i) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    plt.close(fig)
    print(f"已输出：{out_pdf_i}")

print("完成：每个 valid_time 已分别输出为单独 PDF，并已修复 0 度拼接白线。")
