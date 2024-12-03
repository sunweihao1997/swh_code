'''
2024-11-23
This script is to do postprocess of the downloaded hourly ERA5 data
'''
import xarray as xr
import numpy as np
import os

path_in = '/home/sun/mydown/ERA5/era5_psl_u10_v10/'
path_out = '/home/sun/mydown/ERA5/era5_psl_u10_v10_daily/'

file_list = os.listdir(path_in)

ref_file  = xr.open_dataset(path_in + file_list[5])

time      = ref_file.valid_time.data[0]
date = time.astype('datetime64[Y]').astype(int) + 1970

file_list_nc = [x for x in file_list if '.nc' in x]
#print(len(file_list_nc))

for ff in file_list_nc:
    #1. get the year
    f0 = xr.open_dataset(path_in + ff)
    time      = f0.valid_time.data[0]
    date = time.astype('datetime64[Y]').astype(int) + 1970

    print(f'Now it is dealing with {date}')

    f1 = f0.resample(valid_time='24H').sum(dim='valid_time')

    f1.to_netcdf(path_out + str(date) + '.nc')