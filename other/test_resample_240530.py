import xarray as xr

f = xr.open_dataset("/home/sun/mydown/ERA5/era5_precipitation/ERA5_single_hourly_10u_10v_prect_1965.nc")

a = f.tp.resample(time='24H')

print(a)