'''
2024-6-18
This script is for the regression calculation, here I hope to do pre-process for the data by extracting single month 200 hPa layer
'''
import xarray as xr
import numpy as np
import os
from scipy.stats import pearsonr

# ================= File Information ==================
data_path = "/home/sun/mydown/ERA5/monthly_pressure/"

ref_file  = xr.open_dataset(data_path + "ERA5_2000_monthly_pressure_UTVWZSH.nc").sel(level=slice(700, 1000))
lat       = ref_file.latitude.data
lon       = ref_file.longitude
level     = ref_file.level.data

findex    = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_mar.nc")

# =====================================================

def extract_function(varname, start_yyyy, end_yyyy, month0):
    '''
        Only extract single month
    '''
    # 1. Claim the array
    array0 = np.zeros((end_yyyy - start_yyyy + 1, len(level), len(lat), len(lon)))

    # 2. Put the data into the array
    filename = "ERA5_yyyy_monthly_pressure_UTVWZSH.nc"  

    j = 0
    for yy in np.linspace(start_yyyy, end_yyyy, end_yyyy - start_yyyy + 1, dtype=int):
        filename1 = filename.replace("yyyy", str(yy))
        #print(filename1)

        f1        = xr.open_dataset(data_path + filename1).sel(level=slice(700, 1000))
        f1_month  = f1.sel(time=f1.time.dt.month.isin([month0]))

        array0[j] = f1_month[varname].data

        j         += 1

    return array0

def cal_correlation(series, array):
    '''
        This function is to calculate pearsons correlation between series and given 4d array
    '''
    correlation = np.zeros(array[0].shape)
    p           = np.zeros(array[0].shape)

    for ll in range(len(level)):
        for ii in range(len(lat)):
            for jj in range(len(lon)):
                correlation[ll, ii, jj], p[ll, ii, jj] = pearsonr(series, array[:, ll, ii, jj])

    return correlation, p




if __name__ == '__main__':
    start0 = 1980 ; end0 = 2021
    # 1. u
    Mar_u = extract_function('u', start0, end0, 3)
    Apr_u = extract_function('u', start0, end0, 4)

    # 2. v 
    Mar_v = extract_function('v', start0, end0, 3)
    Apr_v = extract_function('v', start0, end0, 4)

    # 5. q 
    Mar_q = extract_function('q', start0, end0, 3)
    Apr_q = extract_function('q', start0, end0, 4)

    # 4. Write to ncfile
    # Save to the ncfile
    ncfile  =  xr.Dataset(
    {
        "Mar_qu":    (["time", "lat", "lon"], np.nanmean(Mar_q, axis=1)*np.nanmean(Mar_u, axis=1)/9.8),
        "Apr_qu":    (["time", "lat", "lon"], np.nanmean(Apr_q, axis=1)*np.nanmean(Apr_u, axis=1)/9.8),
        "Mar_qv":    (["time", "lat", "lon"], np.nanmean(Mar_q, axis=1)*np.nanmean(Mar_v, axis=1)/9.8),
        "Apr_qv":    (["time", "lat", "lon"], np.nanmean(Apr_q, axis=1)*np.nanmean(Apr_v, axis=1)/9.8),
    },
    coords={
        "time": (["time"], np.linspace(start0, end0, end0 - start0 + 1)), 
        "lat":  (["lat"],  ref_file.latitude.data),
        "lon":  (["lon"],  ref_file.longitude.data),
    },
    )

    ncfile.attrs['description']  =  'Created on 2024-6-18 by cal_Anomaly_onset_preprocess_qV_240618.py'
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/monthly/ERA5_1980_2021_monthly_March_April_1000_700_qV.nc")