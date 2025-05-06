'''
2023-10-05
This script is to calculate JJA mean in each year

2023-11-13 modified:
changed to calculate JJAS mean

2023-12-21 modified:
change the method to extract the JJA/JJAS months

2024-5-2 update:
calculate single month for JJAS
'''
import xarray as xr
import numpy as np
from netCDF4 import Dataset

file_BTAL    = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_precipitation_ensemble_mean_231005.nc')
file_BTALnEU = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/noEU_precipitation_ensemble_mean_231004.nc')

# === Claim the array saving the result ===
#print(len(file0['time'].data)/12) #156 Years. Here I only calculate 157 years, and the begin is 1850-01
JJAS_prect_BTAL    = np.zeros((157, 4, 96, 144))
JJAS_prect_BTALnEU = np.zeros((157, 4, 96, 144))

# EXtract the months
month_JJAS= [6, 7, 8, 9]

j = 0 # number for month loop
for month0 in month_JJAS:
    #print(file_BTAL.time.dt.month)
    file_single_month_BTAL    = file_BTAL.sel(time=file_BTAL.time.dt.month.isin([month0]))
    file_single_month_BTALnEU = file_BTALnEU.sel(time=file_BTALnEU.time.dt.month.isin([month0]))

    JJAS_prect_BTAL[:, j, ]     =  file_single_month_BTAL['PRECC'].data  + file_single_month_BTAL['PRECL'].data
    JJAS_prect_BTALnEU[:, j, ]  =  file_single_month_BTALnEU['PRECC'].data  + file_single_month_BTALnEU['PRECL'].data

    j += 1

## === Write to ncfile ===
ncfile  =  xr.Dataset(
{
    "JJAS_prect_BTAL":       (["year", "month", "lat", "lon"], JJAS_prect_BTAL     * 86400000),
    "JJAS_prect_BTALnEU":    (["year", "month", "lat", "lon"], JJAS_prect_BTALnEU  * 86400000),
},
coords={
    "year": (["year"], np.linspace(1850, 1850+156, 157)),
    "month":(["month"],month_JJAS),
    "lat":  (["lat"],  file_BTAL['lat'].data),
    "lon":  (["lon"],  file_BTAL['lon'].data),
},
)

ncfile['JJAS_prect_BTAL'].attrs['units']     = 'mm day^-1'
ncfile['JJAS_prect_BTALnEU'].attrs['units']    = 'mm day^-1'

ncfile.attrs['description']  =  'Created on 2024-5-2. This file is the update of previous result which save the value for each month in JJAS.'

ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_single_month_JJAS_1850_2006.nc", format='NETCDF4')