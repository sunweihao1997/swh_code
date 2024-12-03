'''
2024-4-1
This script is to used daily-averaged SST to calculate multiple years pentad and month mean Nino34 index
'''
import xarray as xr
import numpy as np

data_path = '/home/sun/data/ERA5_SST/pentad/'
data_name = 'ERA5_pentad_month_SST_1959-2021.nc'

f0        = xr.open_dataset(data_path + data_name).sel(lat=slice(-5, 5), lon=slice(190, 240))

#print(f0)
pentad_nino34 = np.zeros((63, 73))
month_nino34  = np.zeros((63, 12))

# Start calculation
for i in range(63):
    for pp in range(73):
        pentad_nino34[i, pp] = np.nanmean(f0['pentad_sst'].data[i, pp])
    
    for mm in range(12):
        month_nino34[i, mm]  = np.nanmean(f0['month_sst'].data[i, mm])

# Save file to netCDF
ncfile  =  xr.Dataset(
    {
        "pentad_nino34": (["year", "pentad",], pentad_nino34),
        "month_nino34":  (["year", "month",],  month_nino34),
    },
    coords={
        "year":  (["year"],  np.linspace(1959, 2021, 63)),
        "month": (["month"], np.linspace(1, 12, 12)),
        "pentad":(["pentad"],np.linspace(1, 73, 73)),
    },
    )

ncfile.attrs["description"]  =  "Created on 2024-4-1 at ubuntu(Beijing), this file generated from cal_ERA5_sst_Nino34_pentad_month_240401.py and is the pentad-averaged and monthly averaged Nino34 index, unit is K"
ncfile.to_netcdf("/home/sun/data/process/ERA5/ERA5_SST_Nino34_pentad_month_1959_2021.nc")