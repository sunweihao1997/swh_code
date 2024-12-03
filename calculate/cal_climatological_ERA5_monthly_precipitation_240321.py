'''
2024-3-21
This script is to calculate the climatological precipitation using ERA5

The time-resolution is monthly, daily
'''
import xarray as xr
import numpy as np
import os
import cftime

src_path = '/home/sun/mydown/swh_era5/land/daily/'

# The full file list
file_list = os.listdir(src_path) ; file_list.sort()

# Filter out the precipitation
file_precip = []
for ff in file_list:
    if 'total_precipitation' in ff:
        file_precip.append(ff)

# Claim the array for climatology
pr_day = np.zeros((365, 181, 360))

# The range of the time
year_start = 1980 ; year_end = 2014

for yyyy in range(year_start, year_end + 1):
    file_name = str(int(yyyy)) + '_total_precipitation_land_monthly.nc'

    f1        = xr.open_dataset(src_path + file_name)

    pr_day    += f1['tp'].data[:365] / (year_end - year_start + 1)

dates_climate = cftime.num2date(range(1, 366), units='days since 2000-01-01', calendar='standard')

# Write to ncfile
ncfile  =  xr.Dataset(
        {
            'pr_climate': (["time", "lat", "lon"], pr_day),
                    },
                    coords={
                        "lat": (["lat"],   ref_file.lat.data),
                        "lon": (["lon"],   ref_file.lon.data),
                        "time": (["time"], dates_climate),
                    },
                    )
ncfile['pr_climate'].attrs = f1['pr'].attrs

ncfile.attrs['description'] = 'created on 2024-3-21. This is f2000 ensemble hybrid-10 experiment output. I calculated pentad average based on the daily climate variables'

            ncfile.to_netcdf('/home/sun/data/model_data/f2000_ensemble/hybrid-2/pentad/'+nnnn+'_pentad.nc')