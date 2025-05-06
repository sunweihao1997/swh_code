'''
2025-3-24
This script is to calculate the JJA mean SST for BTAL experiment
'''
import xarray as xr
import numpy as np
from netCDF4 import Dataset

file0 = xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_SST_ensemble_mean_231020.nc')

# === Claim the array saving the result ===
#print(len(file0['time'].data)/12) #156 Years. Here I only calculate 150 years, and the begin is 1850-01
jjas_prect = np.zeros((157, 96, 144))

data_JJAS = file0.sel(time=file0.time.dt.month.isin([7, 8, 9,]))

# === Calculation ===
for i in range(157):
    #a = np.average(file0['PRECC'].data[first_mon : first_mon + 3], axis=0)
    #print(a.shape)
    jjas_prect[i, :, :]  =  np.average(data_JJAS['SST'].data[i*3 : i*3 + 3], axis=0)

# === Write to ncfile ===
ncfile  =  xr.Dataset(
{
    "SST_JJA": (["time", "lat", "lon"], jjas_prect),
},
coords={
    "time": (["time"], np.linspace(1850, 1850+156, 157)),
    "lat":  (["lat"],  file0['lat'].data),
    "lon":  (["lon"],  file0['lon'].data),
},
)

ncfile['SST_JJA'].attrs = file0['SST'].attrs


ncfile.attrs['description']  =  'Created on 2025-3-24 by /home/sun/uoe-code/calculate/cal_jja_sst_each_year_BTAL_250324.py. This file is the JJA ensemble mean among the 8 member in the BTAL emission experiments. The variables is SST. This file is the corrected version which modify the time-axis'
ncfile.to_netcdf("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_SST_jja_mean.nc", format='NETCDF4')