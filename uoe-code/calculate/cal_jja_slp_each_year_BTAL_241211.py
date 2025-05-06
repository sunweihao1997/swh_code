'''
2023-10-05
This script is to calculate JJA mean in each year

241211:
modified the data: change to isin(7, 8, 9) since the model output show one month lag
'''
import xarray as xr
import numpy as np
from netCDF4 import Dataset

file0 = xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_SLP_ensemble_mean_231005.nc')
print(file0.time)

# === Claim the array saving the result ===
#print(len(file0['time'].data)/12) #156 Years. Here I only calculate 150 years, and the begin is 1850-01
jjas_prect = np.zeros((157, 96, 144))

data_JJAS = file0.sel(time=file0.time.dt.month.isin([7, 8, 9]))

# === Calculation ===
for i in range(157):
    #a = np.average(file0['PRECC'].data[first_mon : first_mon + 3], axis=0)
    #print(a.shape)
    jjas_prect[i, :, :]  =  np.average(data_JJAS['SLP'].data[i*3 : i*3 + 3], axis=0)

# === Write to ncfile ===
ncfile  =  xr.Dataset(
{
    "SLP_JJA": (["time", "lat", "lon"], jjas_prect),
},
coords={
    "time": (["time"], np.linspace(1850, 1850+156, 157)),
    "lat":  (["lat"],  file0['lat'].data),
    "lon":  (["lon"],  file0['lon'].data),
},
)

ncfile['SLP_JJA'].attrs = file0['SLP'].attrs


ncfile.attrs['description']  =  'Created on 2024-12-11 by /home/sun/uoe-code/calculate/cal_jja_slp_each_year_BTAL_241211.py. This file is the JJA ensemble mean among the 8 member in the BTAL emission experiments. The variables is SLP. This file is the modified type for the BTAL_SLP_jja_mean_231005.nc'
ncfile.to_netcdf("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_SLP_jja_mean_241211.nc", format='NETCDF4')