'''
2024-4-2
This script is to calculate the Nino34 index based on HadISST dataset
'''
import xarray as xr
import numpy as np
import sys

data_path = '/home/sun/mydown/HadISST/'
data_name = 'HadISST_sst_down_2022-1-24.nc'

fsst      = xr.open_dataset(data_path + data_name)

#print(fsst.time.data[-1]) # End on 2021-10-16
year_list = np.linspace(1940, 2020, 2020-1940+1)

lat_slice = slice(20, -20)
lon_slice = slice(40, 100)

fsst_nino34 = fsst.sel(latitude=lat_slice, longitude=lon_slice)

Nino34_array = np.zeros((len(year_list), 12))

for yy in year_list:
    fsst_nino34_1year = fsst_nino34.sel(time=fsst_nino34.time.dt.year.isin([yy]))

    if len(fsst_nino34_1year.time.data) != 12:
        sys.exit(f'The year {yy} includes {len(fsst_nino34_1year.time.data)} months')

    for mm in range(12):
        Nino34_array[int(yy - 1940), mm] = np.nanmean(fsst_nino34_1year['sst'].data[mm])

    print(f'Finished year {yy}')
    print(Nino34_array[int(yy - 1940),])

# Write to ncfile
# Save file to netCDF
ncfile  =  xr.Dataset(
    {
        "month_iob":  (["year", "month",],  Nino34_array),
    },
    coords={
        "year":  (["year"],  np.linspace(1940, 2020, 2020-1940+1)),
        "month": (["month"], np.linspace(1, 12, 12)),
    },
    )

ncfile.attrs["description"]  =  "Created on 2024-4-2 at ubuntu(Beijing), this file generated from cal_HadISST_Nino34_1940_2022_240402.py and is the monthly averaged Nino34 index"
ncfile.to_netcdf("/home/sun/data/process/HadISST/HadISST_SST_IOB_month_1940_2020.nc")