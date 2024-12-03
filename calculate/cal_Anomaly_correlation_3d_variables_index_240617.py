'''
2024-6-16
This script is for the regression calculation, here I hope to do pre-process for the data by extracting single month 200 hPa layer
'''
import xarray as xr
import numpy as np
import os
from scipy.stats import pearsonr

# ================= File Information ==================
data_path = "/home/sun/mydown/ERA5/monthly_pressure/"

ref_file  = xr.open_dataset(data_path + "ERA5_2000_monthly_pressure_UTVWZSH.nc").sel(level=[150, 200, 300, 400, 500, 700, 850, 925])
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

        f1        = xr.open_dataset(data_path + filename1).sel(level=[150, 200, 300, 400, 500, 700, 850, 925])
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

    #test1, test2  = cal_correlation(findex['OLR_mari_Afri'].data, Mar_u)

    #print(np.max(test1))

    # 2. v 
    Mar_v = extract_function('v', start0, end0, 3)
    Apr_v = extract_function('v', start0, end0, 4)

    # 3. z 
    Mar_z = extract_function('z', start0, end0, 3)
    Apr_z = extract_function('z', start0, end0, 4)

    # 4. w 
    Mar_w = extract_function('w', start0, end0, 3)
    Apr_w = extract_function('w', start0, end0, 4)

    # 5. q 
    Mar_q = extract_function('q', start0, end0, 3)
    Apr_q = extract_function('q', start0, end0, 4)

    # 4. Write to ncfile
    # Save to the ncfile
    ncfile  =  xr.Dataset(
    {
        "Mar_u":     (["time", "level", "lat", "lon"], Mar_u),
        "Apr_u":     (["time", "level", "lat", "lon"], Apr_u),
        "Mar_v":     (["time", "level", "lat", "lon"], Mar_v),
        "Apr_v":     (["time", "level", "lat", "lon"], Apr_v),
        "Mar_z":     (["time", "level", "lat", "lon"], Mar_z),
        "Apr_z":     (["time", "level", "lat", "lon"], Apr_z),
        "Mar_w":     (["time", "level", "lat", "lon"], Mar_w),
        "Apr_w":     (["time", "level", "lat", "lon"], Apr_w),
        "Mar_q":     (["time", "level", "lat", "lon"], Mar_q),
        "Apr_q":     (["time", "level", "lat", "lon"], Apr_q),

        "corre_u_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_u)[0]),
        "corre_u_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_u)[0]),
        "corre_v_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_v)[0]),
        "corre_v_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_v)[0]),
        "corre_z_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_z)[0]),
        "corre_z_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_z)[0]),
        "corre_w_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_w)[0]),
        "corre_w_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_w)[0]),
        "corre_q_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_q)[0]),
        "corre_q_olr":     (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_q)[0]),

        "p_u_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_u)[1]),
        "p_u_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_u)[1]),
        "p_v_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_v)[1]),
        "p_v_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_v)[1]),
        "p_z_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_z)[1]),
        "p_z_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_z)[1]),
        "p_w_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_w)[1]),
        "p_w_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_w)[1]),
        "p_q_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Mar_q)[1]),
        "p_q_olr":         (["level", "lat", "lon"], cal_correlation(findex['OLR_mari_Afri'].data, Apr_q)[1]),

        "corre_u_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_u)[0]),
        "corre_u_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_u)[0]),
        "corre_v_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_v)[0]),
        "corre_v_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_v)[0]),
        "corre_z_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_z)[0]),
        "corre_z_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_z)[0]),
        "corre_w_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_w)[0]),
        "corre_w_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_w)[0]),
        "corre_q_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_q)[0]),
        "corre_q_lstc":     (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_q)[0]),

        "p_u_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_u)[1]),
        "p_u_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_u)[1]),
        "p_v_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_v)[1]),
        "p_v_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_v)[1]),
        "p_z_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_z)[1]),
        "p_z_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_z)[1]),
        "p_w_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_w)[1]),
        "p_w_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_w)[1]),
        "p_q_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Mar_q)[1]),
        "p_q_lstc":         (["level", "lat", "lon"], cal_correlation(findex['LSTC_psl_IOB'].data, Apr_q)[1]),
    },
    coords={
        "time": (["time"], np.linspace(start0, end0, end0 - start0 + 1)),
        "lev":  (["level"], level),
        "lat":  (["lat"],  ref_file.latitude.data),
        "lon":  (["lon"],  ref_file.longitude.data),
    },
    )

    ncfile.attrs['description']  =  'Created on 2024-6-17 by cal_Anomaly_regression_preprocess_200hPa_variables_240616.py.'
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/monthly/ERA5_1980_2021_monthly_3d_UVZWQ.nc")