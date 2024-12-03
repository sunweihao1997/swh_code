'''
2024-5-7
This script is to calculate the area-averaged OLR and analyze its power spectral during April and May
'''
import xarray as xr
import numpy as np
import os

# ============ File Information ==========
data_path = '/home/sun/data/other_data/down_ERA5_hourly_OLR_convert_float_daymean/'

ref_file  = xr.open_dataset(data_path + '1957_hourly_OLR.nc')

#a         = np.average(ref_file['ttr'].data)
#print(a/86400*24)
start_year = 1980 ; end_year = 2019

# ========================================

# Concatenate the files from 1980 to 2019
def concatenate_OLR(filelist):
    multi_year_OLR = []

    for yyyy in filelist:
        f1 = xr.open_dataset(data_path + str(int(yyyy)) + "_hourly_OLR.nc")

        f1_Apr_May = f1.sel(time=f1.time.dt.month.isin([4, 5])).sel(longitude=slice(90, 100), latitude=slice(20, 10))

        multi_year_OLR.append(f1_Apr_May['ttr'].data/86400*24)

    olr_new = np.concatenate(multi_year_OLR, axis=0)

    return olr_new

def cal_daily_anomaly(series0):
    '''
        This function calculate daily anomalies and normalize the result
    '''
    # Filter the data with 5-day moving average
    series0 = np.convolve(series0, np.ones(5), "same") / 5

    #year_number = len(series0) / 61
    #print(year_number)
    for dd in range(61): # 61 is the total days for April and May
        index0 = np.arange(dd, len(series0), 61)

        series0[index0] -= np.average(series0[index0])

        series0[index0] = (series0[index0] - series0[index0].min() )/ (series0[index0].max() - series0[index0].min())

    return series0


if __name__ == '__main__':
    year_list = np.linspace(1980, 2019, 2019-1980+1)

    olr       = concatenate_OLR(year_list)

    BOB_olr   = np.average(np.average(olr, axis=1), axis=1)

    BOB_olr_anomaly = cal_daily_anomaly(BOB_olr)

    # Write to ncfile
    ncfile  =  xr.Dataset(
        {
            "BOB_olr_anomaly": (["time"], BOB_olr_anomaly),
        },
        )

    ncfile.attrs['description']  =  'This file saves the BOB OLR for April and May, the data has been 5-day moving average and normalized, which is for the power spectral analysis.'
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_OLR_Apr_May_normalized.nc")