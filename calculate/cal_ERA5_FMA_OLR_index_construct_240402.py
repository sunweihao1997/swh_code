'''
2024-4-2
This script is to construct Feb-Mar-Apr mean OLR index to do correlationship analysis with monsoon onset
'''
import xarray as xr
import numpy as np

file0  =  xr.open_dataset('/home/sun/data/download/ERA5_OLR_monthly/era5_monthly_OLR_1940-2022.nc')

#print(file0.time.data)
year_list = np.linspace(1940, 2020, 2020-1940+1)

OLR_index = np.zeros((len(year_list)))

for yy in year_list:
    # Extract 1 year
    file0_1year = file0.sel(time=file0.time.dt.year.isin(yy)).sel(latitude=slice(10, -5), longitude=slice(100, 130))

    OLR_index[int(yy) - 1940]   =   np.nanmean(file0_1year['ttr'].data[2:4]) / 3600

    #print(OLR_index[int(yy) - 1940])

# Save to the ncfile
ncfile  =  xr.Dataset(
    {
        "FMA_OLR": (["year",], OLR_index),
    },
    coords={
        "year":  (["year"],  year_list),
    },
    )

ncfile.attrs["description"]  =  "Created on 2024-4-2 at ubuntu(Beijing), this file generated from cal_ERA5_FMA_OLR_index_construct_240402.py and is the area-averaged OLR over maritime continent (-5~10N, 100~130E) for 1940-2020"
ncfile.to_netcdf("/home/sun/data/process/ERA5/ERA5_OLR_Mar-Apr_month_maritime_continent_1940-2020.nc")