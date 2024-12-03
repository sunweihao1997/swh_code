'''
2024-9-19
This script serves for figs1 in ERL paper, in order to calculate climatological JJA ERA5 data for

1. 850 hPa wind 2. Pr
'''
import xarray as xr
import numpy as np
from scipy import stats
from scipy.stats import linregress
import sys
from sklearn.linear_model import LinearRegression
from matplotlib.ticker import FormatStrFormatter

# =================== Path for data ====================
path_pressure = "/home/sun/mydown/ERA5/monthly_pressure/"
path_single   = "/home/sun/mydown/ERA5/monthly_single/"

start_yyyy    = 1980 ; end_yyyy    = 2005 # Which range of climatology
# =====================================================

def cal_single_climate(start, end, path_src, varname, month_range):
    # 1. obtain the file list
    import os

    file_list = os.listdir(path_src)
    file_list = [ff for ff in file_list if ff[-2:] == "nc"] ; file_list.sort()
    #print(file_list)

    # 2. claim the climate array
    ref_file = xr.open_dataset(path_src + file_list[5])
    lat      = ref_file.latitude.data ; lon     = ref_file.longitude.data
    avg_array= np.zeros((len(lat), len(lon)))

    #print(avg_array.shape)

    # 3. Start calculating
    for yyyy in np.linspace(start, end, end - start + 1, dtype=int):
        f1     = xr.open_dataset(path_src + str(yyyy) + "_single_month.nc")
        f1_jja = f1.sel(time=f1.time.dt.month.isin(month_range))

        avg_array += (np.average(f1_jja[varname].data, axis=0) / (end - start + 1))

    ncfile  =  xr.Dataset(
            {
                varname:     (["lat", "lon"], avg_array),  
            },
            coords={
                "lat":  (["lat"],  lat),
                "lon":  (["lon"],  lon),
            },
        )
    
    ncfile[varname].attrs = f1[varname].attrs

    return ncfile

    #print(np.average(avg_array))
    return avg_array

def cal_pressure_climate(start, end, path_src, varname, month_range):
    # 1. obtain the file list
    import os

    file_list = os.listdir(path_src)
    file_list = [ff for ff in file_list if ff[-2:] == "nc"] ; file_list.sort()
    #print(file_list)

    # 2. claim the climate array
    ref_file = xr.open_dataset(path_src + file_list[5])
    lat      = ref_file.latitude.data ; lon     = ref_file.longitude.data ; level     = ref_file.level.data
    avg_array= np.zeros((len(level), len(lat), len(lon)))

    #print(avg_array.shape)

    # 3. Start calculating
    for yyyy in np.linspace(start, end, end - start + 1, dtype=int):
        f1     = xr.open_dataset(path_src + "ERA5_abcd_monthly_pressure_UTVWZSH.nc".replace("abcd", str(yyyy)))
        f1_jja = f1.sel(time=f1.time.dt.month.isin(month_range))

        avg_array += (np.average(f1_jja[varname].data, axis=0) / (end - start + 1))

    ncfile  =  xr.Dataset(
            {
                varname:     (["level", "lat", "lon"], avg_array),  
            },
            coords={
                "level": (["level"],  level),
                "lat":   (["lat"],    lat),
                "lon":   (["lon"],    lon),
            },
        )
    
    ncfile[varname].attrs = f1[varname].attrs

    return ncfile

    #print(np.average(avg_array))
    return avg_array
    


def main():
    # 1. calculate climate precipitation
    pr_jja = cal_single_climate(start_yyyy, end_yyyy,   path_single, 'tp', [6, 7, 8])
    u_jja  = cal_pressure_climate(start_yyyy, end_yyyy, path_pressure, 'u', [6, 7, 8])
    v_jja  = cal_pressure_climate(start_yyyy, end_yyyy, path_pressure, 'v', [6, 7, 8])

    #print(v_jja)

    # 2. Write to ncfile
    merged_ds = xr.merge([pr_jja, u_jja, v_jja])
    print(merged_ds['tp'])

    merged_ds.attrs['description'] = 'Create on 2024-9-19 by cal_ERL_figs1_model_evaluation_850wind_pr_240919.py on ubuntu server. This is ERA5 1980-2005 climate JJA wind and precipitation.'

    merged_ds.to_netcdf("/home/sun/data/process/ERA5/ERA5_monthly_JJA_wind_precip.nc")

if __name__ == '__main__':
    main()
