'''
2024-5-30
This script is to calculate the 8-80 bandpass OLR
'''
import xarray as xr
import numpy as np
import os
from scipy.signal import butter, filtfilt

# ============ File Information ==========
data_path = '/home/sun/data/other_data/down_ERA5_hourly_OLR_convert_float_daymean/'

ref_file  = xr.open_dataset(data_path + '1957_hourly_OLR.nc')

#a         = np.average(ref_file['ttr'].data)
#print(a/86400*24)
start_year = 1980 ; end_year = 2019

# ========================================
def band_pass_calculation(data, fs, low_frq, high_frq, order,):
    '''
        fs: sample freq
    '''
    lowcut  = 1/low_frq
    highcut = 1/high_frq

    b, a    = butter(N=order, Wn=[lowcut, highcut], btype='band', fs=fs)

    filtered_data = filtfilt(b, a, data)

    return filtered_data

def low_pass_calculation(data, fs, low_frq, high_frq, order,):
    '''
        fs: sample freq
    '''
    lowcut  = 1/low_frq
    highcut = 1/high_frq

    b, a    = butter(N=order, Wn=lowcut, btype='high', fs=fs)

    filtered_data = filtfilt(b, a, data)

    return filtered_data
    

# Concatenate the files from 1980 to 2019
def concatenate_OLR(filelist):
    multi_year_OLR = []

    time_new = []

    for yyyy in filelist:
        f1 = xr.open_dataset(data_path + str(int(yyyy)) + "_hourly_OLR.nc")

        f1_Apr_May = f1

        multi_year_OLR.append(f1_Apr_May['ttr'].data[:365]/86400*24) # dump the leap year
        time_new.append(f1_Apr_May['time'].data[:365])

    olr_new = np.concatenate(multi_year_OLR, axis=0)
    time_all= np.concatenate(time_new, axis=0)

    return olr_new, time_all

# Function for bandpass filter


if __name__ == '__main__':
    year_list = np.linspace(1980, 2021, 2021-1980+1)

    olr, time       = concatenate_OLR(year_list)

    for i in range(len(ref_file.latitude.data)):
        for j in range(len(ref_file.longitude.data)):
            olr[:, i, j] = band_pass_calculation(olr[:, i, j], 1, 80, 30, 5)

#    print(olr.shape)

    #print(time.shape)

#    BOB_olr   = np.average(np.average(olr, axis=1), axis=1)


    # Write to ncfile
    ncfile  =  xr.Dataset(
        {
            "olr":  (["time", "lat", "lon"], olr),
        },
        coords=dict(
        time=("time", time),
        lat=("lat", ref_file.latitude.data),
        lon=("lon", ref_file.longitude.data),
    ),
        )

    ncfile.attrs['description']  =  'This file saves the BOB OLR for the 1980-2021, The bandpass is 30-80 days'
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_OLR_bandpass_filter_30_80.nc")