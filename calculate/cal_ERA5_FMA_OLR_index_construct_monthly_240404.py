'''
2024-4-2
This script is to construct Feb-Mar-Apr mean OLR index to do correlationship analysis with monsoon onset

2024-4-4 modified:
calculate monthly index
'''
import xarray as xr
import numpy as np

file0  =  xr.open_dataset('/home/sun/data/download/ERA5_OLR_monthly/era5_monthly_OLR_1940-2022.nc')
maskf  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

#print(file0.time.data)
year_list = np.linspace(1940, 2020, 2020-1940+1)

OLR_index = np.zeros((len(year_list), 12))

for yy in year_list:
    # Extract 1 year
    file0_1year = file0.sel(time=file0.time.dt.year.isin(yy)).sel(latitude=slice(10, -5), longitude=slice(100, 130))
    print(len(file0_1year.time.data))

    #print(OLR_index[int(yy) - 1940])
    for mon in range(12):
        OLR_index[int(yy) - 1940, mon]   =   np.nanmean(file0_1year['ttr'].data[mon]) / 3600

# Save to the ncfile
ncfile  =  xr.Dataset(
    {
        "FMA_OLR": (["year", "month"], OLR_index),
    },
    coords={
        "year":   (["year"],   year_list),
        "month":  (["month"],  np.linspace(1, 12, 12)),
    },
    )

ncfile.attrs["description"]  =  "Created on 2024-4-2 at ubuntu(Beijing), this file generated from cal_ERA5_FMA_OLR_index_construct_monthly_240404.py and is the area-averaged OLR over maritime continent (-5~10N, 100~130E) for 1940-2020"
ncfile.to_netcdf("/home/sun/data/process/ERA5/ERA5_OLR_month_maritime_continent_1940-2020.nc")