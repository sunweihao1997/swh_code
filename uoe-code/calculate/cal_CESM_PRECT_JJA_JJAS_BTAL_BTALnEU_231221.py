'''
2023-10-05
This script is to calculate JJA mean in each year

2023-11-13 modified:
changed to calculate JJAS mean

2023-12-21 modified:
change the method to extract the JJA/JJAS months
'''
import xarray as xr
import numpy as np
from netCDF4 import Dataset

file_BTAL    = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_precipitation_ensemble_mean_231005.nc')
file_BTALnEU = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/noEU_precipitation_ensemble_mean_231004.nc')

# === Claim the array saving the result ===
#print(len(file0['time'].data)/12) #156 Years. Here I only calculate 157 years, and the begin is 1850-01
JJA_prect_BTAL     = np.zeros((157, 96, 144))
JJA_prect_BTALnEU  = np.zeros((157, 96, 144))
JJAS_prect_BTAL    = np.zeros((157, 96, 144))
JJAS_prect_BTALnEU = np.zeros((157, 96, 144))

# EXtract the months
month_JJA = [6, 7, 8]
month_JJAS= [6, 7, 8, 9]

file_BTAL_JJA  = file_BTAL.sel(time=file_BTAL.time.dt.month.isin(month_JJA))
file_BTAL_JJAS = file_BTAL.sel(time=file_BTAL.time.dt.month.isin(month_JJAS))
file_BTALnEU_JJA  = file_BTALnEU.sel(time=file_BTALnEU.time.dt.month.isin(month_JJA))
file_BTALnEU_JJAS = file_BTALnEU.sel(time=file_BTALnEU.time.dt.month.isin(month_JJAS))



# === Calculation ===
for i in range(157):

    JJA_prect_BTAL[i]  =  np.average(file_BTAL_JJA['PRECC'].data[i * 3 : i*3 + 3], axis=0)  + np.average(file_BTAL_JJA['PRECL'].data[i*3 : i*3 + 3], axis=0)
    JJAS_prect_BTAL[i] =  np.average(file_BTAL_JJAS['PRECC'].data[i * 4 : i*4 + 4], axis=0) + np.average(file_BTAL_JJAS['PRECL'].data[i*4 : i*4 + 4], axis=0)
    JJA_prect_BTALnEU[i]  =  np.average(file_BTALnEU_JJA['PRECC'].data[i * 3 : i*3 + 3], axis=0)  + np.average(file_BTALnEU_JJA['PRECL'].data[i*3 : i*3 + 3], axis=0)
    JJAS_prect_BTALnEU[i] =  np.average(file_BTALnEU_JJAS['PRECC'].data[i * 4 : i*4 + 4], axis=0) + np.average(file_BTALnEU_JJAS['PRECL'].data[i*4 : i*4 + 4], axis=0)


# === Write to ncfile ===
ncfile  =  xr.Dataset(
{
    "PRECT_JJA_BTAL":     (["time", "lat", "lon"], JJA_prect_BTAL     * 86400000),
    "PRECT_JJAS_BTAL":    (["time", "lat", "lon"], JJAS_prect_BTAL    * 86400000),
    "PRECT_JJA_BTALnEU":  (["time", "lat", "lon"], JJA_prect_BTALnEU  * 86400000),
    "PRECT_JJAS_BTALnEU": (["time", "lat", "lon"], JJAS_prect_BTALnEU * 86400000),
},
coords={
    "time": (["time"], np.linspace(1850, 1850+156, 157)),
    "lat":  (["lat"],  file_BTAL['lat'].data),
    "lon":  (["lon"],  file_BTAL['lon'].data),
},
)

ncfile['PRECT_JJA_BTAL'].attrs['units']     = 'mm day^-1'
ncfile['PRECT_JJAS_BTAL'].attrs['units']    = 'mm day^-1'
ncfile['PRECT_JJA_BTALnEU'].attrs['units']  = 'mm day^-1'
ncfile['PRECT_JJAS_BTALnEU'].attrs['units'] = 'mm day^-1'
ncfile.attrs['description']  =  'Created on 2023-12-21. This file is the update of previous result which wrongly calculate the JJA/JJAS average.'

ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc", format='NETCDF4')