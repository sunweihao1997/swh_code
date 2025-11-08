import numpy as np
import xarray as xr
from pathlib import Path

# ============ 配置区 ============
# 文件路径（按你给的路径）
GPCC_PATH = "/home/sun/data/download_data/data/analysis_data/Aerosol_Research_GPCC_PRECT_JJA_JJAS_linear_trend_1901to1955.nc"
ALL_PATH  = "/home/sun/data/download_data/data/analysis_data/Aerosol_Research_CESM_BTAL_PRECT_JJA_linear_trend_1901to1955_corrected.nc"

# 若文件内变量名不确定，代码会自动探测；也可手动指定（如 'trend'、'PRECT_trend' 等）
GPCC_VAR = "JJA_trend"
ALL_VAR  = "JJA_trend"

# 用“四个角点”定义矩形框，顺序不严格（代码会自动识别）
# 例：[(lonW, latS), (lonE, latS), (lonE, latN), (lonW, latN)]
# 经度建议给在 -180..180 范围；若给了 0..360 也会自动转换
CORNERS = [
    # 示例：请改成你的四个点（经度、纬度）
    ( 70.0,  10),   # 西南角
    (90.0,  10),   # 东南角
    (90.0, 25.0),   # 东北角
    (70.0, 25.0),   # 西北角
]
# ---------- 工具函数 ----------
def _find_lat_lon_names(ds):
    cand_lat = ["lat", "latitude", "LAT", "nav_lat", "y"]
    cand_lon = ["lon", "longitude", "LON", "nav_lon", "x"]
    lat_name = next((n for n in cand_lat if n in ds.coords), None)
    lon_name = next((n for n in cand_lon if n in ds.coords), None)
    if lat_name is None or lon_name is None:
        # 也可能在 dims 里
        lat_name = lat_name or next((n for n in cand_lat if n in ds.dims), None)
        lon_name = lon_name or next((n for n in cand_lon if n in ds.dims), None)
    if lat_name is None or lon_name is None:
        raise ValueError("未能自动识别 lat/lon 坐标名，请手动检查 NetCDF。")
    return lat_name, lon_name

def _to_minus180_180(lon):
    # 将经度标准化到 [-180, 180)
    lon180 = ((lon + 180) % 360) - 180
    # 处理恰好为 180 的情况，统一到 -180..180)
    lon180 = xr.where(lon180 == -180, 180.0, lon180)
    return lon180

def _standardize_grid(da):
    """重命名为 lat/lon，并把经度转为 [-180,180)，按经纬升序排序"""
    ds = da.to_dataset(name="var") if isinstance(da, xr.DataArray) else da
    lat_name, lon_name = _find_lat_lon_names(ds)

    # 重命名坐标为 lat/lon
    if lat_name != "lat" or lon_name != "lon":
        ds = ds.rename({lat_name: "lat", lon_name: "lon"})

    # 经度到 [-180,180)
    if float(ds.lon.max()) > 180.0 or float(ds.lon.min()) >= 0.0:
        ds = ds.assign_coords(lon=_to_minus180_180(ds.lon))

    # 按经纬排序（升序）
    if (ds.lat[1] - ds.lat[0]).values < 0:
        ds = ds.sortby("lat")
    if (ds.lon[1] - ds.lon[0]).values < 0:
        ds = ds.sortby("lon")

    return ds

def _pick_data_var(ds, prefer=None):
    """选一个包含 lat/lon 的变量；若给了 prefer 就优先取"""
    if prefer and prefer in ds.data_vars:
        da = ds[prefer]
    else:
        candidates = [v for v in ds.data_vars]
        if not candidates:
            raise ValueError("数据集中不存在 data_vars。")
        # 选第一个包含 lat/lon 维的变量
        picked = None
        for v in candidates:
            dims = set(ds[v].dims)
            if "lat" in dims and "lon" in dims:
                picked = v; break
        if picked is None:
            raise ValueError("未找到同时含 lat/lon 维度的变量，请指定变量名。")
        da = ds[picked]

    # 如果还带 time 等无关维，尽量 squeeze 掉长度为1的维
    for dim in list(da.dims):
        if dim not in ("lat","lon") and da.sizes[dim] == 1:
            da = da.squeeze(dim)
    # 再次确认只剩 lat/lon
    if not (("lat" in da.dims) and ("lon" in da.dims)):
        raise ValueError(f"所选变量 {da.name} 不是纯二维(lat/lon)；请预处理或指定变量。")
    return da

def _bbox_from_corners(corners):
    """从四个角点推断矩形；自动判断是否跨经线。返回 (lon_min, lon_max, lat_min, lat_max)，
       若跨日界线，则令 lon_min > lon_max 来表示。"""
    if len(corners) == 0:
        return None
    lons = np.array([c[0] for c in corners], dtype=float)
    lats = np.array([c[1] for c in corners], dtype=float)

    # 统一经度到 [-180,180)
    lons = ((lons + 180) % 360) - 180
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))

    # 两种经度包络的宽度：直接宽度 vs 跨日界线宽度
    direct_min, direct_max = float(np.min(lons)), float(np.max(lons))
    width_direct = direct_max - direct_min
    width_cross  = 360.0 - width_direct

    if width_cross < width_direct:
        # 选择跨日界线的更“小”的矩形：用 lon_min > lon_max 表示跨界
        lon_min = float(np.max(lons))
        lon_max = float(np.min(lons))
    else:
        lon_min, lon_max = direct_min, direct_max

    return lon_min, lon_max, lat_min, lat_max

def _subset_rect(da, lon_min, lon_max, lat_min, lat_max):
    """
    用二维布尔掩膜按矩形裁剪；不依赖经纬坐标单调性。
    约定：在 [-180,180) 经度系下，若 lon_min > lon_max 表示跨 180° 经线。
    """
    # 广播 1D lat/lon 成 2D 网格
    lon2d, lat2d = xr.broadcast(da.lon, da.lat)

    # 纬度条件（不关心坐标升/降序）
    if lat_min <= lat_max:
        lat_mask = (lat2d >= lat_min) & (lat2d <= lat_max)
    else:
        lat_mask = (lat2d >= lat_max) & (lat2d <= lat_min)

    # 经度条件：跨界与否分别处理
    if lon_min <= lon_max:
        lon_mask = (lon2d >= lon_min) & (lon2d <= lon_max)
    else:
        # 跨 180°：左段 OR 右段
        lon_mask = (lon2d >= lon_min) | (lon2d <= lon_max)

    mask = lat_mask & lon_mask
    # drop=True 丢掉完全为 NaN 的行列，返回紧致子域
    return da.where(mask, drop=True)


def area_weighted_pearson(x, y):
    """面积加权 Pearson（权重 = cos(lat)），x,y 为同网格 DataArray"""
    # 广播权重到 2D
    w_lat = np.cos(np.deg2rad(x.lat))
    w = w_lat.broadcast_like(x)

    # 有效掩膜
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    xv = x.where(mask).values.ravel()
    yv = y.where(mask).values.ravel()
    wv = w.where(mask).values.ravel()

    m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(wv) & (wv > 0)
    xv, yv, wv = xv[m], yv[m], wv[m]
    if xv.size == 0:
        raise ValueError("有效格点数为 0，请检查掩膜或裁剪区域。")

    # 加权去中心
    mx = np.average(xv, weights=wv)
    my = np.average(yv, weights=wv)
    xv = xv - mx
    yv = yv - my

    cov = np.average(xv * yv, weights=wv)
    vx  = np.average(xv * xv, weights=wv)
    vy  = np.average(yv * yv, weights=wv)
    r = float(cov / np.sqrt(vx * vy))

    # 同时给个幅度信息（可选）
    stdx = np.sqrt(vx)
    stdy = np.sqrt(vy)
    amp_ratio = float(stdy / stdx)  # y/x 幅度比

    # 粗略有效样本量（Kish），仅反映权重不均衡
    Neff = float((wv.sum() ** 2) / (np.sum(wv ** 2)))
    # Fisher z 95% CI（保守的，因为未计入空间自相关）
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(Neff - 3.0, 1.0))
    ci = tuple(np.tanh([z - 1.96 * se, z + 1.96 * se]))

    return r, amp_ratio, Neff, ci


# ---------- 主流程 ----------
def main():
    # 读取并标准化网格
    gpcc_ds = xr.open_dataset(GPCC_PATH)
    all_ds  = xr.open_dataset(ALL_PATH)

    gpcc_ds = _standardize_grid(gpcc_ds)
    all_ds  = _standardize_grid(all_ds)

    gpcc = _pick_data_var(gpcc_ds, GPCC_VAR)  # 2D(lat,lon)
    allm = _pick_data_var(all_ds,  ALL_VAR)   # 2D(lat,lon) 或已 squeeze

    # 将 ALL 插值到 GPCC 网格
    all_on_gpcc = allm.interp(lat=gpcc.lat, lon=gpcc.lon, method="linear")

    # 用 GPCC 的 NaN 做共同掩膜
    mask = np.isfinite(gpcc)
    gpcc_m = gpcc.where(mask)
    all_m  = all_on_gpcc.where(mask)

    # 若给定四角点，则做区域裁剪（支持跨日界线）
    bbox = _bbox_from_corners(CORNERS)
    if bbox is not None:
        lon_min, lon_max, lat_min, lat_max = bbox
        gpcc_m = _subset_rect(gpcc_m, lon_min, lon_max, lat_min, lat_max)
        all_m  = _subset_rect(all_m,  lon_min, lon_max, lat_min, lat_max)

    # 计算面积加权 Pearson
    r, amp_ratio, Neff, ci = area_weighted_pearson(gpcc_m, all_m)

    # 输出
    if bbox is None:
        region_msg = "（全域共同有效区域）"
    else:
        region_msg = f"（区域：lon {lon_min}→{lon_max}，lat {lat_min}→{lat_max}；跨界用 lon_min>lon_max 表示）"

    print(f"[面积加权 Pearson] r = {r:.3f}  95% CI [{ci[0]:.3f}, {ci[1]:.3f}]  Neff≈{Neff:.0f} {region_msg}")
    print(f"[幅度比] ALL/GPCC 的加权标准差比 = {amp_ratio:.3f}  （<1 表示 ALL 幅度偏弱）")

if __name__ == "__main__":
    main()