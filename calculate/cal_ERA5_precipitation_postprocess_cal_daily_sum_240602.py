'''
2024-6-2
This script is to calculate the daily sum for the ERA5 data
'''
import xarray as xr
import numpy as np
import os

path_in = '/home/sun/mydown/ERA5/era5_precipitation/'

path_out = '/home/sun/mydown/ERA5/era5_precipitation_daily/'

file_list = os.listdir(path_in) ; file_list.sort()

for ff in file_list:
    if ff[0] != '.' and ff[-2:] == 'nc':
        f0 = xr.open_dataset(path_in + ff)

        f1 = f0.resample(time='24H').sum(dim='time')

        f1.to_netcdf(path_out + ff)