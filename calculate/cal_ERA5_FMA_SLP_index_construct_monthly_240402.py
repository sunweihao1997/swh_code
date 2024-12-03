'''
2024-4-2
This script is to construct Feb-Mar-Apr mean SLP index to do correlationship analysis with monsoon onset

here the SLP index means meridional gradient between 70-90E

2024-4-4
'''
import xarray as xr
import numpy as np

file0  =  xr.open_dataset('/home/sun/data/download/ERA5_SLP_monthly/era5_monthly_slp_1940_2022.nc')
maskf  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

#print(file0.time.data)
year_list = np.linspace(1940, 2020, 2020-1940+1)

SLP_index = np.zeros((len(year_list), 12))

for yy in year_list:
    # Extract 1 year
    file0_1year_land = file0.sel(time=file0.time.dt.year.isin(yy)).sel(latitude=slice(20, 5), longitude=slice(70, 90))
    mask_land        = maskf.sel(latitude=slice(20, 5), longitude=slice(70, 90))
    slp_land         = file0_1year_land['msl'].data

    print(len(file0_1year_land.time.data))
    # Mask Sea for Indian 
#    for tt in range(slp_land.shape[0]):
#        slp_land[tt][mask_land['lsm'].data[0] < 0.1] = np.nan
    

    file0_1year_sea  = file0.sel(time=file0.time.dt.year.isin(yy)).sel(latitude=slice(20,  0), longitude=slice(40, 100))
    mask_sea         = maskf.sel(latitude=slice(20, 0), longitude=slice(40, 100))
    slp_sea          = file0_1year_sea['msl'].data
    #Mask land for IOB region
    for tt in range(slp_sea.shape[0]):
        slp_sea[tt][mask_sea['lsm'].data[0] > 0.1] = np.nan

#    zon_slp     = np.average(np.average(file0_1year['msl'].data[1:4], axis=0), axis=1)

#    SLP_index[int(yy) - 1940] = np.nanmean(slp_land[1:4]) - np.nanmean(slp_sea[1:4])
    for mon in range(12):
        SLP_index[int(yy) - 1940, mon] = np.nanmean(slp_land[mon])

    #print(SLP_index[int(yy) - 1940])

# Save to the ncfile
ncfile  =  xr.Dataset(
    {
        "FMA_SLP": (["year","month"], SLP_index),
    },
    coords={
        "year":   (["year"],   year_list),
        "month":  (["month"],  np.linspace(1, 12, 12)),
    },
    )

ncfile.attrs["description"]  =  "Created on 2024-4-4 at ubuntu(Beijing), this file generated from cal_ERA5_FMA_SLP_index_construct_monthly_240402.py and is the area-averaged SLP only over the Indian Continent"
ncfile.to_netcdf("/home/sun/data/process/ERA5/ERA5_SLP_month_land_slp_70-90_1940-2020.nc")