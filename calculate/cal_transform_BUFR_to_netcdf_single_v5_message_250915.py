#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

# ======================= 配置（按需修改） =======================
IN_DIR           = "/home/sun/data/bufr_test"                 # 单 message 的 nc 文件目录
OUT_DIR          = "/home/sun/data/bufr_merged_1hourwindow"               # 输出目录
RES_DEG          = 0.5                                       # 格点分辨率（度）。例：1.0 / 0.5 / 0.25
TIME_WINDOW_MIN  = 60                                       # 时间聚合窗口（分钟）。None = 不做时间维（全时混合）；例如 60 = 1 小时
MAX_FILES_PER_MT = None                                       # 测试用：每个 msg_type 最多处理多少个文件；None = 全部
# ===============================================================


def find_msg_types(in_dir: str) -> list[str]:
    """从文件名 message_XXXXX_<msgtype>.nc中识别所有 msg_type。"""
    types = set()
    for f in glob.glob(os.path.join(in_dir, "*.nc")):
        base = os.path.basename(f)
        # 兼容 message_01234_NC005012.nc 或 message_00001_ADPUPA.nc 这类命名
        parts = base.rsplit("_", 1)
        if len(parts) == 2 and parts[0].startswith("message_"):
            mt = parts[1].removesuffix(".nc")
            types.add(mt)
    return sorted(types)


def load_all_for_type(in_dir: str, msg_type: str, max_files: int | None = None) -> pd.DataFrame:
    """读取某个 msg_type 的所有单 message 文件 → 拼成一个大 DataFrame（obs表）"""
    pattern = os.path.join(in_dir, f"*_{msg_type}.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"未找到 {msg_type} 的文件：{pattern}")
    if max_files is not None:
        files = files[:max_files]

    dfs = []
    for f in files:
        ds = xr.load_dataset(f)

        # 必要变量校验 & 经纬度兜底（CLAT/CLON 或 CLATH/CLONH）
        has_clat = "CLAT" in ds
        has_clon = "CLON" in ds
        lat = (ds[("CLAT" if has_clat else "CLATH")].values).astype(float)
        lon = (ds[("CLON" if has_clon else "CLONH")].values).astype(float)

        for k in ["WDIR", "WSPD", "time"]:
            if k not in ds:
                raise RuntimeError(f"{f} 缺少变量 {k}")

        df = pd.DataFrame({
            "time": pd.to_datetime(ds["time"].values),
            "lat":  lat,
            "lon":  lon,
            "wdir": ds["WDIR"].values.astype(float),
            "wspd": ds["WSPD"].values.astype(float),
        })
        dfs.append(df)

    big = pd.concat(dfs, ignore_index=True)

    # 清理缺测，归一化经度到 [-180, 180)
    big = big.replace([np.inf, -np.inf], np.nan).dropna(subset=["lat", "lon", "wdir", "wspd"])
    big["lon"] = ((big["lon"] + 180.0) % 360.0) - 180.0

    return big


# ---------- 风向/风速 与 U/V 的互转（气象风向：来自方向；北=0°，顺时针） ----------
def met_wind_to_uv(wdir_deg: np.ndarray, wspd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    th = np.deg2rad(wdir_deg)
    u = -wspd * np.sin(th)
    v = -wspd * np.cos(th)
    return u, v

def uv_to_met_wind(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wspd = np.sqrt(u*u + v*v)
    # 去向角 = atan2(v,u)；来自角 = 去向角 + 180°
    to_deg = (np.degrees(np.arctan2(v, u)) + 360.0) % 360.0
    wdir = (to_deg + 180.0) % 360.0
    return wdir, wspd
# ----------------------------------------------------------------


def maybe_time_bin(df: pd.DataFrame, window_min: int | None) -> tuple[pd.DataFrame, bool]:
    """时间分箱：None=不分箱（无time维），否则 floor到窗口起始（产生 time 维）。"""
    if window_min is None:
        out = df.copy()
        out["tbin"] = pd.NaT
        return out, False
    out = df.copy()
    out["tbin"] = out["time"].dt.floor(f"{int(window_min)}min")
    return out, True


def grid_bin_average(df: pd.DataFrame, res_deg: float, with_time: bool) -> xr.Dataset:
    """按规则网格箱均值（bin-averaging）：先U/V平均，再回到WDIR/WSPD。"""
    # 先转U/V，避免直接平均风向造成的圆周偏差
    u, v = met_wind_to_uv(df["wdir"].values, df["wspd"].values)
    df = df.assign(u=u, v=v)

    # 网格边界与中心
    lat_edges = np.arange(-90,  90 + res_deg, res_deg)
    lon_edges = np.arange(-180, 180 + res_deg, res_deg)
    lat_cent  = (lat_edges[:-1] + lat_edges[1:]) / 2.0
    lon_cent  = (lon_edges[:-1] + lon_edges[1:]) / 2.0

    # 落箱
    df["ilat"] = pd.cut(df["lat"], bins=lat_edges, labels=False, include_lowest=True)
    df["ilon"] = pd.cut(df["lon"], bins=lon_edges, labels=False, include_lowest=True)
    df = df.dropna(subset=["ilat", "ilon"])
    df["ilat"] = df["ilat"].astype(int)
    df["ilon"] = df["ilon"].astype(int)

    # 分组键
    keys = ["ilat", "ilon"] if not with_time else ["tbin", "ilat", "ilon"]

    # 箱均值（U/V）与计数
    grp = df.groupby(keys, dropna=False).agg(
        u_mean=("u", "mean"),
        v_mean=("v", "mean"),
        nobs=("u", "size"),
    ).reset_index()

    # 预分配与填充
    if with_time:
        times = pd.to_datetime(sorted(grp["tbin"].dropna().unique()))
        U = xr.DataArray(np.full((times.size, lat_cent.size, lon_cent.size), np.nan, "float32"),
                         coords={"time": times, "lat": lat_cent, "lon": lon_cent},
                         dims=("time", "lat", "lon"))
        V = U.copy(deep=True); N = U.copy(deep=True)
        idx_time = {t: i for i, t in enumerate(times)}
        for _, r in grp.iterrows():
            i, j = int(r["ilat"]), int(r["ilon"])
            if pd.isna(r["tbin"]):   # 正常不会出现
                continue
            ti = idx_time[pd.to_datetime(r["tbin"])]
            U.values[ti, i, j] = r["u_mean"]
            V.values[ti, i, j] = r["v_mean"]
            N.values[ti, i, j] = r["nobs"]
    else:
        U = xr.DataArray(np.full((lat_cent.size, lon_cent.size), np.nan, "float32"),
                         coords={"lat": lat_cent, "lon": lon_cent},
                         dims=("lat", "lon"))
        V = U.copy(deep=True); N = U.copy(deep=True)
        for _, r in grp.iterrows():
            i, j = int(r["ilat"]), int(r["ilon"])
            U.values[i, j] = r["u_mean"]
            V.values[i, j] = r["v_mean"]
            N.values[i, j] = r["nobs"]

    # 回到风向/风速
    WDIR, WSPD = uv_to_met_wind(U.values, V.values)

    ds_out = xr.Dataset(
        data_vars=dict(
            U=(U.dims, U.values),
            V=(V.dims, V.values),
            WDIR=(U.dims, WDIR.astype("float32")),
            WSPD=(U.dims, WSPD.astype("float32")),
            NOBS=(U.dims, N.values.astype("float32")),
        ),
        coords=U.coords,
        attrs=dict(
            title="Gridded winds (bin-averaged via U/V)",
            grid_res_deg=str(res_deg),
            time_binning=("none" if not with_time else f"{int(TIME_WINDOW_MIN)}min"),
        )
    )
    return ds_out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) 扫描 msg_type
    msg_types = find_msg_types(IN_DIR)
    if not msg_types:
        raise SystemExit("未在目录中发现任何 .nc 文件。")

    print("发现的 msg_type：", ", ".join(msg_types))

    # 2) 逐个 msg_type 处理
    for mt in msg_types:
        print(f"\n==> 处理 {mt}")
        # 2.1 读取并拼接（obs表）
        df = load_all_for_type(IN_DIR, mt, MAX_FILES_PER_MT)

        # 2.2 保存合并后的 obs 表（NetCDF）
        ds_obs = xr.Dataset(
            data_vars=dict(
                CLAT=("obs", df["lat"].values.astype("float32")),
                CLON=("obs", df["lon"].values.astype("float32")),
                WDIR=("obs", df["wdir"].values.astype("float32")),
                WSPD=("obs", df["wspd"].values.astype("float32")),
                time=("obs", df["time"].values.astype("datetime64[ns]")),
            ),
            attrs=dict(msg_type=mt)
        )
        merged_path = os.path.join(OUT_DIR, f"{mt}_merged.nc")
        ds_obs.to_netcdf(merged_path)
        print(f"[OK] 合并输出：{merged_path}  (nobs={ds_obs.dims['obs']})")

        # 2.3 时间分箱（决定是否产生 time 维）
        df_binned, with_time = maybe_time_bin(df, TIME_WINDOW_MIN)

        # 2.4 格点化（箱均值）
        ds_grid = grid_bin_average(df_binned, RES_DEG, with_time)
        ds_grid.attrs["msg_type"] = mt

        gridded_path = os.path.join(OUT_DIR, f"{mt}_gridded.nc")
        ds_grid.to_netcdf(gridded_path)
        if "time" in ds_grid.dims:
            print(f"[OK] 格点输出：{gridded_path}  (time={ds_grid.dims['time']}, lat={ds_grid.dims['lat']}, lon={ds_grid.dims['lon']})")
        else:
            print(f"[OK] 格点输出：{gridded_path}  (lat={ds_grid.dims['lat']}, lon={ds_grid.dims['lon']})")


if __name__ == "__main__":
    main()
