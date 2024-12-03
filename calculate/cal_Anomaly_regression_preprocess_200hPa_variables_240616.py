'''
2024-6-16
This script is for the regression calculation, here I hope to do pre-process for the data by extracting single month 200 hPa layer
'''
import xarray as xr
import numpy as np
import os

# ================= File Information ==================
data_path = "/home/sun/mydown/ERA5/monthly_pressure/"

ref_file  = xr.open_dataset(data_path + "ERA5_2000_monthly_pressure_UTVWZSH.nc")
lat       = ref_file.latitude.data
lon       = ref_file.longitude
level     = ref_file.level.data

# =====================================================

def extract_function(varname, level, start_yyyy, end_yyyy, month0):
    '''
        Only extract single month
    '''
    # 1. Claim the array
    array0 = np.zeros((end_yyyy - start_yyyy + 1, len(lat), len(lon)))

    # 2. Put the data into the array
    filename = "ERA5_yyyy_monthly_pressure_UTVWZSH.nc"  

    j = 0
    for yy in np.linspace(start_yyyy, end_yyyy, end_yyyy - start_yyyy + 1, dtype=int):
        filename1 = filename.replace("yyyy", str(yy))
        #print(filename1)

        f1        = xr.open_dataset(data_path + filename1).sel(level=level)
        f1_month  = f1.sel(time=f1.time.dt.month.isin([month0]))

        array0[j] = f1_month[varname].data

        j         += 1

    return array0




if __name__ == '__main__':
    start0 = 1980 ; end0 = 2021
    # 1. March u 200 hPa
    Mar_u = extract_function('u', 200, start0, end0, 3)
    Apr_u = extract_function('u', 200, start0, end0, 4)

    # 2. v 200hPa
    Mar_v = extract_function('v', 200, start0, end0, 3)
    Apr_v = extract_function('v', 200, start0, end0, 4)

    # 3. z 200hPa
    Mar_z = extract_function('z', 200, start0, end0, 3)
    Apr_z = extract_function('z', 200, start0, end0, 4)

    # 4. w 500hPa
    Mar_w = extract_function('w', 500, start0, end0, 3)
    Apr_w = extract_function('w', 500, start0, end0, 4)

    # 5. T 200hPa
    Mar_t = extract_function('t', 200, start0, end0, 3)
    Apr_t = extract_function('t', 200, start0, end0, 4)

    # 4. Write to ncfile
    # Save to the ncfile
    ncfile  =  xr.Dataset(
    {
        "Mar_u":     (["time", "lat", "lon"], Mar_u),
        "Apr_u":     (["time", "lat", "lon"], Apr_u),
        "Mar_v":     (["time", "lat", "lon"], Mar_v),
        "Apr_v":     (["time", "lat", "lon"], Apr_v),
        "Mar_z":     (["time", "lat", "lon"], Mar_z),
        "Apr_z":     (["time", "lat", "lon"], Apr_z),
        "Mar_w":     (["time", "lat", "lon"], Mar_w),
        "Apr_w":     (["time", "lat", "lon"], Apr_w),
        "Mar_t":     (["time", "lat", "lon"], Mar_t),
        "Apr_t":     (["time", "lat", "lon"], Apr_t),
    },
    coords={
        "time": (["time"], np.linspace(start0, end0, end0 - start0 + 1)),
        "lat":  (["lat"],  ref_file.latitude.data),
        "lon":  (["lon"],  ref_file.longitude.data),
    },
    )

    ncfile.attrs['description']  =  'Created on 2024-6-17 by cal_Anomaly_regression_preprocess_200hPa_variables_240616.py. 2024-7-16 update: Add T variable'
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/monthly/ERA5_1980_2021_monthly_200hpa_UVZ_500hpa_w.nc")