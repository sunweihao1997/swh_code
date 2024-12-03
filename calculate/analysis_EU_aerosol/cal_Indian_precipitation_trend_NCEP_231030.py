'''
2023-10-30
This script is to calculate the long-term trend in precipitation over Indian
'''
import xarray as xr
import numpy as np

file_path = '/home/sun/mydown/NCEP_prect/'
file_name = 'precip.mon.total.1x1.v2020.nc'

# Calculate the JJA mean or JJAS mean
m0 = 5 # June
m1 = 7 # August

# Calim the array
season_rain = np.zeros((129, 180, 360))

f0 = xr.open_dataset(file_path + file_name)

for yyyy in range(129):
    season_rain[yyyy] = np.average(f0['precip'].data[yyyy * 12 + m0:yyyy * 12 + m1], axis=0)

# Save the result to the Dataset
ncfile  =  xr.Dataset(
                    {
                        'seasonal_precipitation': (["time","lat","lon"], season_rain),
                    },
                    coords={
                        "time": (["time"], np.linspace(1891,1891 + 129, 129)),
                        "lat":  (["lat"],  f0.lat.data),
                        "lon":  (["lon"],  f0.lon.data),
                    },
                    )
ncfile.attrs['description'] = 'created on 2023-10-30. This file save the seasonal rainfall (month total) from NCEP precipitation from 1891-2020'
ncfile['seasonal_precipitation'].attrs = f0['precip'].attrs

ncfile.to_netcdf('/home/sun/data/long_term_precipitation/Precipitation_single_NCEP_1x1_1891_2020_20231030.nc')