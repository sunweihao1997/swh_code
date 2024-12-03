'''
2024-7-3
This script is to calculate climatological variables using MERRA2, serving as data for Phd thesis figure 4.1

variables include u, v, psl, t
'''
import xarray as xr
import numpy as np
import os

# ========== File Information ==========
path_src = "/data1/MERRA2/daily/plev/yyyy/"

ref_file = xr.open_dataset("/data1/MERRA2/daily/plev/1980/MERRA2_100.inst3_3d_asm_Np.19800413.SUB.nc").sel(lev=slice(1000, 500))
lat      = ref_file.lat.data ; lon       = ref_file.lon.data ; lev      = ref_file.lev.data

# ======================================


def cal_climate_var(varname, start_yyyy, end_yyyy):
    # 1. Claim the average array
    avg_array = np.zeros((365, len(lat), len(lon)))

    # 2. years number
    num_year  = end_yyyy - start_yyyy + 1

    # 3. Start calculation
    for dddd in range(365):
        print(f"Now it is dealing with {varname}, the day {dddd}")
        for yy in np.linspace(start_yyyy, end_yyyy, end_yyyy - start_yyyy + 1, dtype=int):
            path1 = path_src.replace("yyyy", str(yy))

            file_list1 = os.listdir(path1) ; file_list1.sort()

            f1         = xr.open_dataset(path1 + file_list1[dddd]).sel(lev=925)

            avg_array[dddd] += (f1[varname].data[0] / num_year)

    return avg_array

def cal_climate_var_3d(varname, start_yyyy, end_yyyy):
    # 1. Claim the average array
    avg_array = np.zeros((365, len(lev), len(lat), len(lon)))

    # 2. years number
    num_year  = end_yyyy - start_yyyy + 1

    # 3. Start calculation
    for dddd in range(365):
        print(f"Now it is dealing with {varname}, the day {dddd}")
        for yy in np.linspace(start_yyyy, end_yyyy, end_yyyy - start_yyyy + 1, dtype=int):
            path1 = path_src.replace("yyyy", str(yy))

            file_list1 = os.listdir(path1) ; file_list1.sort()

            f1         = xr.open_dataset(path1 + file_list1[dddd]).sel(lev=slice(1000, 500))

            avg_array[dddd] += (f1[varname].data[0] / num_year)

    return avg_array

if __name__ == "__main__":
#    slp = cal_climate_var("SLP", 1980, 2019)
#    v925= cal_climate_var("V",   1980, 2019)
#    u925= cal_climate_var("U",   1980, 2019)
    v    = cal_climate_var_3d("V",   1980, 2019)

    # ----------- save to the ncfile ------------------
    ncfile  =  xr.Dataset(
    {
        "v": (["time", "lev", "lat", "lon"], v),
    },
    coords={
        "time": (["time"], np.linspace(1, 365, 365)),
        "lat":  (["lat"],  lat),
        "lon":  (["lon"],  lon),
        "lev":  (["lev"],  lev),
    },
    )
    ncfile.attrs['description']  =  'MERRA2 climatological (1980-2019) daily variables'
    ncfile.to_netcdf("/data4/2019swh/data/MERRA2/MERRA2_climate_daily_v.nc")