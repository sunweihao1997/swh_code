'''
231005
This script calculate FSDS among 8 ensemble experiments noEU
'''
import xarray as xr
import numpy as np
import os

path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/FSDS/'

ref_file  = xr.open_dataset(path0 + 'BTAL_FSDS_1850_150years_member_1.nc')
#print(ref_file)

# === Claim the array for mean ===
prect_c = np.zeros((1883, 96, 144)) ; prect_l = np.zeros((1883, 96, 144))

# === Calculate ensemble mean ===
for i in range(8):
    f0 = xr.open_dataset(path0 + 'BTAL_FSDS_1850_150years_member_' + str(i + 1) +'.nc')

    prect_c += f0['FSDS'].data/8

print('Success')

# === Write these data to nc file ===
# ----------- save to the ncfile ------------------
ncfile  =  xr.Dataset(
{
    "FSDS": (["time", "lat", "lon"], prect_c),
},
coords={
    "time": (["time"], ref_file['time'].data),
    "lat":  (["lat"],  ref_file['lat'].data),
    "lon":  (["lon"],  ref_file['lon'].data),
},
)

ncfile['FSDS'].attrs = ref_file['FSDS'].attrs

ncfile.attrs['description']  =  'Created on 2023-10-05. This file is the ensemble mean among the 8 member in the noEU emission experiments. The variable is FSDS'
ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTAL_FSDS_ensemble_mean_231005.nc", format='NETCDF4')
