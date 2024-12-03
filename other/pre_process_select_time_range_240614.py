'''
2024-6-14
This script is to screen out the 1980-2021 year for LSTC data
'''
import xarray as xr
import numpy as np

f0  =  xr.open_dataset("/home/sun/mydown/ERA5/monthly_single/multiple_year_single_vars.nc")

f1  =  f0.sel(time=f0.time.dt.year.isin(np.linspace(1980, 2021, 42)))

f1['msl'].to_netcdf("/home/sun/mydown/ERA5/monthly_single/1980_2021_multiple_year_single_psl.nc")