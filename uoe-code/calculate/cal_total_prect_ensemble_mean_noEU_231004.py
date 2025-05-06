'''
231004
This script calculate total precipitation among 8 ensemble experiments noEU
'''
import xarray as xr
import numpy as np
import os

path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/PRECT/'

ref_file  = xr.open_dataset(path0 + 'noEU_PRECC_1850_150years_member_1.nc')
ref_file2 = xr.open_dataset(path0 + 'noEU_PRECL_1850_150years_member_1.nc')
#print(ref_file)

# === Claim the array for mean ===
prect_c = np.zeros((1883, 96, 144)) ; prect_l = np.zeros((1883, 96, 144))

# === Calculate ensemble mean ===
for i in range(8):
    f0 = xr.open_dataset(path0 + 'noEU_PRECC_1850_150years_member_' + str(i + 1) +'.nc')
    f1 = xr.open_dataset(path0 + 'noEU_PRECL_1850_150years_member_' + str(i + 1) +'.nc')

    prect_c += f0['PRECC'].data/8
    prect_l += f1['PRECL'].data/8

print('Success')

# === Write these data to nc file ===
# ----------- save to the ncfile ------------------
ncfile  =  xr.Dataset(
{
    "PRECC": (["time", "lat", "lon"], prect_c),
    "PRECL": (["time", "lat", "lon"], prect_l),
},
coords={
    "time": (["time"], ref_file['time'].data),
    "lat":  (["lat"],  ref_file['lat'].data),
    "lon":  (["lon"],  ref_file['lon'].data),
},
)

ncfile['PRECC'].attrs = ref_file['PRECC'].attrs
ncfile['PRECL'].attrs = ref_file2['PRECL'].attrs

ncfile.attrs['description']  =  'Created on 2023-10-04. This file is the ensemble mean among the 8 member in the noEU emission experiments. The variables are PRECC and PRECL, and their sum is total precipitation.'
ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/noEU_precipitation_ensemble_mean_231004.nc", format='NETCDF4')
