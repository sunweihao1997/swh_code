'''
2023-8-26
This script calculate pentad-avg OLR time series from 1940 to 2022, according to Prof.Wu's advise, here I calculate zonal diff OLR between western Indian Ocean and Maritime Continent
The select area is 100-130E, 0-10N, the second area I select 30-45E, 0-10N
'''
import os
import xarray as xr
import numpy as np

path0 = '/home/sun/data/other_data/down_ERA5_hourly_OLR_convert_float_pentadmean/'
file_list = os.listdir(path0) ; file_list.sort()
pentad_mean_olr = np.zeros((83 ,73))

year_n = 0
for ffff in file_list:
    f0 = xr.open_dataset(path0 + ffff).sel(latitude=slice(10, 0), longitude=slice(100, 130))
    f1 = xr.open_dataset(path0 + ffff).sel(latitude=slice(10, 0), longitude=slice(30, 45))

    for tttt in range(73):
        pentad_mean_olr[year_n, tttt] = np.average(f0['ttr'].data[tttt]/3600 * -1) - np.average(f1['ttr'].data[tttt]/3600 * -1)

    year_n += 1

ncfile  =  xr.Dataset(
{
    "olr_diff": (["year", "pentad"], pentad_mean_olr),
},
coords={
    "year": (["year"], np.linspace(1940, 2022, 83)),
    "pentad": (["pentad"], np.linspace(1, 73, 73)),
},
)
ncfile['olr_diff'].attrs['units'] = 'W m**-2'
ncfile.attrs['description']  =  'It is the pentad mean OLR difference between the maritime continent(0-10N, 100-130E) and eastern Africa(0-10N, 30-45E) from 1940 to 2022'
ncfile.to_netcdf("/home/sun/data/long_time_series_after_process/ERA5/ERA5_OLR_diff_maritime_continent_eastern_Africa_1940_2022.nc")