"""
2024-6-4
This script is to calculate the LSTC index using ERA5 data

MODIFIED (2026-01-18):
- Disable per-year data_path workflow: ALL calculations read from a single SLP file
- Remove OLR indices
- Save land_mean and ocean_mean components in output NetCDF
- Paths: /home/sun/data -> /home/sun/wd_14/data_beijing
"""

import numpy as np
import xarray as xr
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ================= Paths ==========================
slp_file = "/home/sun/wd_14/data_beijing/download/ERA5_SLP_monthly/era5_monthly_slp_1940_2022.nc"
mask_file = "/home/sun/wd_14/data_beijing/mask/ERA5_land_sea_mask.nc"

out_path = "/home/sun/wd_14/data_beijing/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_psl_components.nc"

# ================ Regions =========================
lon_range_land = slice(70, 90)
lat_range_land = slice(20, 5)

# 7090 ocean box
lon_range_ocean_7090 = slice(70, 90)
lat_range_ocean_7090 = slice(20, -20)

# IOB ocean box
lon_range_ocean_iob = slice(40, 100)
lat_range_ocean_iob = slice(20, -20)

# analysis period
START_DATE = "1980-01-01"
END_DATE   = "2021-12-31"

# land/sea threshold
LSM_THR = 0.1

# ==================================================
mask_ds = xr.open_dataset(mask_file)

def _masked_area_mean(da: xr.DataArray, lsm: xr.DataArray, keep_land: bool, thr: float = 0.1) -> xr.DataArray:
    """
    Spatial mean after masking by land/sea mask.
    keep_land=True  -> keep lsm >= thr
    keep_land=False -> keep lsm <= thr
    """
    da2 = da.where(lsm >= thr) if keep_land else da.where(lsm <= thr)
    return da2.mean(dim=("latitude", "longitude"), skipna=True)

def calculate_lstc_components_from_dataset(
    ds: xr.Dataset,
    land_lon, land_lat,
    ocean_lon, ocean_lat,
    varname: str,
    thr: float = 0.1,
):
    """
    Return (lstc, land_mean, ocean_mean) as DataArray, each along time.
    lstc = land_mean - ocean_mean
    """
    da_land  = ds[varname].sel(latitude=land_lat,  longitude=land_lon)
    da_ocean = ds[varname].sel(latitude=ocean_lat, longitude=ocean_lon)

    # mask has a time dimension in your old file; use the first slice
    lsm_land  = mask_ds["lsm"].isel(time=0).sel(latitude=land_lat,  longitude=land_lon)
    lsm_ocean = mask_ds["lsm"].isel(time=0).sel(latitude=ocean_lat, longitude=ocean_lon)

    land_mean  = _masked_area_mean(da_land,  lsm_land,  keep_land=True,  thr=thr)
    ocean_mean = _masked_area_mean(da_ocean, lsm_ocean, keep_land=False, thr=thr)

    lstc = land_mean - ocean_mean
    return lstc, land_mean, ocean_mean

if __name__ == "__main__":

    # 1) read single SLP file
    with xr.open_dataset(slp_file) as ds_all:

        # 2) subset time to 1980-2021
        ds = ds_all.sel(time=slice(START_DATE, END_DATE))

        # 3) compute LSTC based on sea level pressure (variable name is "msl")
        lstc_7090, land_7090, ocean_7090 = calculate_lstc_components_from_dataset(
            ds,
            lon_range_land, lat_range_land,
            lon_range_ocean_7090, lat_range_ocean_7090,
            varname="msl",
            thr=LSM_THR
        )

        lstc_iob, land_iob, ocean_iob = calculate_lstc_components_from_dataset(
            ds,
            lon_range_land, lat_range_land,
            lon_range_ocean_iob, lat_range_ocean_iob,
            varname="msl",
            thr=LSM_THR
        )

        # 4) build output time coordinate
        # Use the dataset's own time coordinate (recommended)
        out_time = ds["time"].data

    # 5) write to NetCDF
    out = xr.Dataset(
        data_vars={
            "LSTC_psl_7090":       (["time"], lstc_7090.data),
            "LSTC_psl_7090_land":  (["time"], land_7090.data),
            "LSTC_psl_7090_ocean": (["time"], ocean_7090.data),

            "LSTC_psl_IOB":        (["time"], lstc_iob.data),
            "LSTC_psl_IOB_land":   (["time"], land_iob.data),
            "LSTC_psl_IOB_ocean":  (["time"], ocean_iob.data),
        },
        coords={"time": ("time", out_time)},
    )

    out.attrs["description"] = (
        "LSTC (land-sea thermal/pressure contrast proxy) computed from ERA5 mean sea level pressure (msl). "
        "Period: 1980-01 to 2021-12. "
        "LSTC = land_mean - ocean_mean. Land_mean and ocean_mean components are saved. "
        "Land box: (20N-5N, 70E-90E). "
        "Ocean box (7090): (20N-20S, 70E-90E). "
        "Ocean box (IOB): (20N-20S, 40E-100E). "
        "Land/sea masking uses ERA5 lsm threshold 0.1. "
        f"Input SLP file: {slp_file}."
    )

    out.to_netcdf(out_path)
    print(f"Saved: {out_path}")
