#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial pattern correlation (area-weighted Pearson) between GPCC and CESM-ALL
linear trends after applying user-specified scaling factors.

Pipeline:
- Load GPCC & ALL
- Interpolate ALL to GPCC grid
- Subset by a four-corner box (supports dateline crossing)
- Snap box edges to nearest GPCC grid points (optional, for reproducibility)
- Mask to common valid cells (both finite)
- Apply scaling: GPCC * 55, ALL * (55 * 86400000)
- Compute area-weighted Pearson r with cos(lat) weights
- 90% two-sided CI using Fisher-z + Kish effective sample size
- Diagnostics & stability warning when variance ~ 0
"""

import numpy as np
import xarray as xr

# ============================ CONFIG ============================

# File paths
GPCC_PATH = "/home/sun/data/download_data/data/analysis_data/Aerosol_Research_GPCC_PRECT_JJA_JJAS_linear_trend_1901to1955.nc"
ALL_PATH  = "/home/sun/data/download_data/data/analysis_data/Aerosol_Research_CESM_BTAL_PRECT_JJA_linear_trend_1901to1955_corrected.nc"

# Data variable names (leave None to auto-detect a 2D (lat,lon) variable)
GPCC_VAR = None
ALL_VAR  = None

# Box by four corners: [(lon,lat), ...]. Use [-180,180) or [0,360) longitudes.
# Leave [] to compute over the common valid domain (no subsetting).
CORNERS = [
    (70.0, 10.0),   # SW
    (90.0, 10.0),   # SE
    (90.0, 30.0),   # NE
    (70.0, 30.0),   # NW
]

# Interpolation method for ALL -> GPCC grid: "linear" or "nearest"
INTERP_METHOD = "linear"

# Two-sided confidence level for CI (0.90/0.95/0.99)
CONF_LEVEL = 0.90

# Also compute area-weighted Spearman (rank) correlation
DO_SPEARMAN = False

# Snap the box edges to the nearest GPCC grid points (recommended for reproducibility)
SNAP_TO_GRID = True

# Warn when scaled ALL weighted std is below this threshold (near-constant field → unstable r)
EPS_STD = 1e-8

# *** Your explicit scaling factors ***
# GPCC_scaled = GPCC_raw * GPCC_MULT
# ALL_scaled  = ALL_raw  * ALL_MULT
GPCC_MULT = 55.0
ALL_MULT  = 55.0 * 86400000.0  # 55 * (86400 * 1000)

# ================================================================


# --------------------- Helper functions -------------------------

def _find_lat_lon_names(ds):
    cand_lat = ["lat", "latitude", "LAT", "nav_lat", "y"]
    cand_lon = ["lon", "longitude", "LON", "nav_lon", "x"]
    lat_name = next((n for n in cand_lat if n in ds.coords), None)
    lon_name = next((n for n in cand_lon if n in ds.coords), None)
    if lat_name is None or lon_name is None:
        lat_name = lat_name or next((n for n in cand_lat if n in ds.dims), None)
        lon_name = lon_name or next((n for n in cand_lon if n in ds.dims), None)
    if lat_name is None or lon_name is None:
        raise ValueError("Could not identify latitude/longitude coordinate names.")
    return lat_name, lon_name

def _to_minus180_180(lon):
    lon180 = ((lon + 180) % 360) - 180
    return xr.where(lon180 == -180, 180.0, lon180)

def _standardize_grid(obj):
    """Rename to lat/lon, convert lon to [-180,180), sort ascending, drop duplicate coords."""
    ds = obj.to_dataset(name="__var__") if isinstance(obj, xr.DataArray) else obj
    lat_name, lon_name = _find_lat_lon_names(ds)
    if lat_name != "lat" or lon_name != "lon":
        ds = ds.rename({lat_name: "lat", lon_name: "lon"})
    if float(ds.lon.max()) > 180.0 or float(ds.lon.min()) >= 0.0:
        ds = ds.assign_coords(lon=_to_minus180_180(ds.lon))
    # sort ascending
    if ds.lat.size > 1 and (ds.lat[1] - ds.lat[0]).values < 0:
        ds = ds.sortby("lat")
    if ds.lon.size > 1 and (ds.lon[1] - ds.lon[0]).values < 0:
        ds = ds.sortby("lon")
    # drop duplicate coordinates if any
    ds = ds.isel(lon=~ds.lon.to_pandas().duplicated())
    ds = ds.isel(lat=~ds.lat.to_pandas().duplicated())
    return ds

def _pick_data_var(ds, prefer=None):
    """Pick a (lat,lon) variable; prefer `prefer` if provided."""
    if prefer and prefer in ds.data_vars:
        da = ds[prefer]
    else:
        picked = None
        for v in ds.data_vars:
            if "lat" in ds[v].dims and "lon" in ds[v].dims:
                picked = v; break
        if picked is None:
            raise ValueError("No (lat,lon) variable found. Please set GPCC_VAR / ALL_VAR.")
        da = ds[picked]
    # squeeze 1-length dims other than lat/lon
    for dim in list(da.dims):
        if dim not in ("lat","lon") and da.sizes.get(dim, 1) == 1:
            da = da.squeeze(dim)
    if not (("lat" in da.dims) and ("lon" in da.dims)):
        raise ValueError(f"Variable {da.name} is not 2D (lat,lon).")
    return da

def _bbox_from_corners(corners):
    """Return (lon_min, lon_max, lat_min, lat_max).
       If crossing dateline, lon_min > lon_max indicates wrap."""
    if not corners:
        return None
    lons = np.array([c[0] for c in corners], dtype=float)
    lats = np.array([c[1] for c in corners], dtype=float)
    lons = ((lons + 180) % 360) - 180
    lat_min, lat_max = float(np.min(lats)), float(np.max(lats))
    direct_min, direct_max = float(np.min(lons)), float(np.max(lons))
    width_direct = direct_max - direct_min
    width_cross  = 360.0 - width_direct
    if width_cross < width_direct:
        lon_min = float(np.max(lons))
        lon_max = float(np.min(lons))
    else:
        lon_min, lon_max = direct_min, direct_max
    return lon_min, lon_max, lat_min, lat_max

def _subset_rect(da, lon_min, lon_max, lat_min, lat_max):
    """Subset by 2D mask (robust to non-monotonic coords)."""
    lon2d, lat2d = xr.broadcast(da.lon, da.lat)
    # latitude mask
    if lat_min <= lat_max:
        lat_mask = (lat2d >= lat_min) & (lat2d <= lat_max)
    else:
        lat_mask = (lat2d >= lat_max) & (lat2d <= lat_min)
    # longitude mask (handle dateline)
    if lon_min <= lon_max:
        lon_mask = (lon2d >= lon_min) & (lon2d <= lon_max)
    else:
        lon_mask = (lon2d >= lon_min) | (lon2d <= lon_max)
    mask = lat_mask & lon_mask
    return da.where(mask, drop=True)

def _nearest_val(coord, val):
    """Return nearest coordinate value w.r.t. val (no monotonicity assumption)."""
    arr = np.asarray(coord.values, dtype=float)
    if coord.name == "lon":
        val = ((val + 180.0) % 360.0) - 180.0
    idx = int(np.nanargmin(np.abs(arr - val)))
    return float(arr[idx])

def _snap_bounds_to_grid(g, lon_min, lon_max, lat_min, lat_max):
    """Snap bounds to nearest GPCC grid points for reproducibility."""
    lon_min_s = _nearest_val(g.lon, lon_min)
    lon_max_s = _nearest_val(g.lon, lon_max)
    lat_min_s = _nearest_val(g.lat, lat_min)
    lat_max_s = _nearest_val(g.lat, lat_max)
    return lon_min_s, lon_max_s, lat_min_s, lat_max_s

def _zcrit_two_sided(conf=0.90):
    """Return z* for two-sided CI (no SciPy dependency)."""
    table = {
        0.80: 1.2815515655446004,
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }
    ks = np.array(list(table.keys()))
    key = float(ks[np.argmin(np.abs(ks - conf))])
    return table[key]

def weighted_std_da(a):
    """Area-weighted std with cos(lat)."""
    w_lat = np.cos(np.deg2rad(a.lat))
    w = w_lat.broadcast_like(a)
    m = np.isfinite(a) & np.isfinite(w) & (w > 0)
    v = a.where(m).values.ravel()
    wv = w.where(m).values.ravel()
    good = np.isfinite(v) & np.isfinite(wv) & (wv > 0)
    v, wv = v[good], wv[good]
    if v.size == 0:
        return np.nan
    mu = np.average(v, weights=wv)
    var = np.average((v - mu) ** 2, weights=wv)
    return float(np.sqrt(var))

def area_weighted_pearson(x, y, conf=CONF_LEVEL):
    """Area-weighted Pearson r with cos(lat) weights. Returns (r, amp_ratio, Neff, CI)."""
    w_lat = np.cos(np.deg2rad(x.lat))
    w = w_lat.broadcast_like(x)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    xv = x.where(mask).values.ravel()
    yv = y.where(mask).values.ravel()
    wv = w.where(mask).values.ravel()
    m = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(wv) & (wv > 0)
    xv, yv, wv = xv[m], yv[m], wv[m]
    if xv.size == 0:
        raise ValueError("No valid grid cells after masking/subsetting.")
    # weighted de-mean
    mx = np.average(xv, weights=wv)
    my = np.average(yv, weights=wv)
    xv = xv - mx
    yv = yv - my
    cov = np.average(xv * yv, weights=wv)
    vx  = np.average(xv * xv, weights=wv)
    vy  = np.average(yv * yv, weights=wv)
    r = float(cov / np.sqrt(vx * vy))
    # amplitude ratio
    stdx = np.sqrt(vx)
    stdy = np.sqrt(vy)
    amp_ratio = float(stdy / stdx)
    # Kish effective N
    Neff = float((wv.sum() ** 2) / (np.sum(wv ** 2)))
    # Fisher z CI
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1.0 / np.sqrt(max(Neff - 3.0, 1.0))
    zcrit = _zcrit_two_sided(conf)
    ci = tuple(np.tanh([z - zcrit * se, z + zcrit * se]))
    return r, amp_ratio, Neff, ci

def area_weighted_spearman(x, y, conf=CONF_LEVEL):
    """Area-weighted Spearman: rank-transform then area-weighted Pearson."""
    def rank_array(a):
        v = a.values.ravel()
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty_like(v, dtype=float)
        ranks[order] = np.arange(1, v.size + 1, dtype=float)
        # average ranks for ties
        uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        cum = np.cumsum(cnt); start = cum - cnt + 1
        mean_rank = (start + cum) / 2.0
        ranks = mean_rank[inv]
        return xr.DataArray(ranks.reshape(a.shape), coords=a.coords, dims=a.dims)
    Rx, Ry = rank_array(x), rank_array(y)
    r_s, _, Neff, ci = area_weighted_pearson(Rx, Ry, conf=conf)
    return r_s, Neff, ci

def _step(arr):
    return float(np.median(np.diff(arr))) if arr.size > 1 else np.nan


# -------------------------- Main -------------------------------

def main():
    # Read & standardize
    gpcc_ds = _standardize_grid(xr.open_dataset(GPCC_PATH))
    all_ds  = _standardize_grid(xr.open_dataset(ALL_PATH))

    # Pick variables
    gpcc = _pick_data_var(gpcc_ds, GPCC_VAR)
    allm = _pick_data_var(all_ds,  ALL_VAR)

    # Interpolate ALL to GPCC grid
    all_on_gpcc = allm.interp(lat=gpcc.lat, lon=gpcc.lon, method=INTERP_METHOD)

    # Build box & optionally snap to grid
    bbox = _bbox_from_corners(CORNERS)
    if bbox is not None:
        lon_min, lon_max, lat_min, lat_max = bbox
        if SNAP_TO_GRID:
            lon_min, lon_max, lat_min, lat_max = _snap_bounds_to_grid(gpcc, lon_min, lon_max, lat_min, lat_max)
        gpcc_sub = _subset_rect(gpcc,        lon_min, lon_max, lat_min, lat_max)
        allg_sub = _subset_rect(all_on_gpcc, lon_min, lon_max, lat_min, lat_max)
        region_msg = f"（区域：lon {lon_min}→{lon_max}，lat {lat_min}→{lat_max}；跨界用 lon_min>lon_max 表示）"
    else:
        gpcc_sub, allg_sub = gpcc, all_on_gpcc
        region_msg = "（全域共同有效区域）"

    # Common mask: both finite (on raw values)
    common_mask = np.isfinite(gpcc_sub) & np.isfinite(allg_sub)
    gpcc_m_raw = gpcc_sub.where(common_mask)
    all_m_raw  = allg_sub.where(common_mask)

    # Diagnostics (before scaling)
    print("[Diag] lat range:", float(gpcc_m_raw.lat.min()), "→", float(gpcc_m_raw.lat.max()), " step≈", _step(gpcc_m_raw.lat))
    print("[Diag] lon range:", float(gpcc_m_raw.lon.min()), "→", float(gpcc_m_raw.lon.max()), " step≈", _step(gpcc_m_raw.lon))
    n_valid = int(np.isfinite(gpcc_m_raw).sum())
    print("[Diag] valid grid cells (both finite):", n_valid)
    print("[Before scaling] GPCC units:", gpcc_m_raw.attrs.get("units"))
    print("[Before scaling]  ALL units:",  all_m_raw.attrs.get("units"))
    print("[Before scaling] GPCC range:", float(gpcc_m_raw.min()), float(gpcc_m_raw.max()))
    print("[Before scaling]  ALL range:", float(all_m_raw.min()),  float(all_m_raw.max()))

    # Apply your explicit scaling
    gpcc_m = gpcc_m_raw * GPCC_MULT
    all_m  = all_m_raw  * ALL_MULT
    gpcc_m.attrs["units"] = f"({gpcc_m_raw.attrs.get('units','')}) * {GPCC_MULT}"
    all_m.attrs["units"]  = f"({all_m_raw.attrs.get('units','')})  * {ALL_MULT}"

    # Amplitudes after scaling
    std_gpcc = weighted_std_da(gpcc_m)
    std_all  = weighted_std_da(all_m)
    amp_ratio0 = std_all / max(std_gpcc, 1e-30)
    print(f"[Scaling applied] GPCC_MULT={GPCC_MULT:g}  ALL_MULT={ALL_MULT:g}")
    print(f"[After scaling]  GPCC range:", float(gpcc_m.min()), float(gpcc_m.max()))
    print(f"[After scaling]   ALL range:", float(all_m.min()),  float(all_m.max()))
    print(f"[Weighted std]    GPCC={std_gpcc:.4e}  ALL={std_all:.4e}  ratio(ALL/GPCC)={amp_ratio0:.4e}")

    if np.isnan(std_all) or std_all < EPS_STD:
        print(f"[Warn] ALL weighted std ≈ {std_all:.3e} (< {EPS_STD}) → r 对微小扰动非常敏感；"
              "建议检查变量/季节/单位或扩大区域/改用更稳健指标。")

    # Pearson (90% CI)
    r, amp_ratio, Neff, ci = area_weighted_pearson(gpcc_m, all_m, conf=CONF_LEVEL)
    print(f"[面积加权 Pearson] r = {r:.3f}  {int(CONF_LEVEL*100)}% CI [{ci[0]:.3f}, {ci[1]:.3f}]  Neff≈{Neff:.0f} {region_msg}")
    print(f"[幅度比] ALL/GPCC 的加权标准差比 = {amp_ratio:.3f}  （科学计数={amp_ratio:.3e}）")

    # Optional: Spearman
    if DO_SPEARMAN:
        rho, Neff_s, ci_s = area_weighted_spearman(gpcc_m, all_m, conf=CONF_LEVEL)
        print(f"[面积加权 Spearman] rho = {rho:.3f}  {int(CONF_LEVEL*100)}% CI [{ci_s[0]:.3f}, {ci_s[1]:.3f}]  Neff≈{Neff_s:.0f}")

if __name__ == "__main__":
    main()
