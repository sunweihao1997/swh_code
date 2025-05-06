'''
2023-10-04
This script is to calculate JJA mean in each year

2023-11-13 modified:
changed to calculate JJAS mean
'''
import xarray as xr
import numpy as np
from netCDF4 import Dataset

file0 = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/noEU_precipitation_ensemble_mean_231004.nc')

# === Claim the array saving the result ===
#print(len(file0['time'].data)/12) #156 Years. Here I only calculate 157 years, and the begin is 1850-01
jja_prect = np.zeros((157, 96, 144))

# === Calculation ===
for i in range(157):
    first_mon  =  i * 12 + 5

    #a = np.average(file0['PRECC'].data[first_mon : first_mon + 3], axis=0)
    #print(a.shape)
    jja_prect[i, :, :]  =  np.average(file0['PRECC'].data[first_mon : first_mon + 4], axis=0) + np.average(file0['PRECL'].data[first_mon : first_mon + 4], axis=0)

# === Write to ncfile ===
outpath = '/exports/csce/datastore/geos/users/s2618078/'
#outpath = '/home/s2618078/'
#ncfile = Dataset(outpath + 'noEU_precipitation_ensemble_mean_JJA_231004.nc', 'w', format='NETCDF3_CLASSIC')
#ncfile.createDimension('time', len(np.linspace(1850, 1850 + 149, 157)))
#ncfile.createDimension('lon',  len(file0['lon'].data))
#ncfile.createDimension('lat',  len(file0['lat'].data))
#
#lonvar  = ncfile.createVariable('lon','float32',('lon')); lonvar[:] = file0['lon'].data
#latvar  = ncfile.createVariable('lat','float32',('lat')); latvar[:] = file0['lat'].data
#timevar = ncfile.createVariable('time','float64',('time')) ; timevar[:] = np.linspace(1850, 1850 + 149, 157)
#prect   = ncfile.createVariable('prect_jja','float32',('time','lat','lon')) ; prect[:] = jja_prect
#
#ncfile.close()
ncfile  =  xr.Dataset(
{
    "PRECT_JJAS": (["time", "lat", "lon"], jja_prect * 86400000),
},
coords={
    "time": (["time"], np.linspace(1850, 1850+156, 157)),
    "lat":  (["lat"],  file0['lat'].data),
    "lon":  (["lon"],  file0['lon'].data),
},
)

ncfile['PRECT_JJAS'].attrs = file0['PRECC'].attrs

#for yyyy in range(157):
#    print(np.average(ncfile["PRECT_JJAS"].data[yyyy]) * 86400000)


ncfile.attrs['description']  =  'Created on 2023-11-13. This file is the JJAS ensemble mean among the 8 member in the noEU emission experiments. The variables are PRECT total'
ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/noEU_precipitation_jjas_mean_231113.nc", format='NETCDF4')