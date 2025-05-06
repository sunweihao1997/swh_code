'''
2024-9-21
This script is to calculate the linear trend of precip from 1901-1955 using Udel dataset
'''
import xarray as xr
import numpy as np

data_path = "/home/sun/data/download_data/UDEL/"
data_name = "precip.mon.total.v501.nc"

f0        = xr.open_dataset(data_path + data_name)

#print(f0) # monthly data from 1900 to 2017
# ========= 1. calculate the JJA mean and write to ncfile =============
lat = f0.lat.data ; lon = f0.lon.data

# 1.1 Claim the array
pr_jja  = np.zeros((118, len(lat), len(lon))) # 1900 to 2017 total 118 years
pr_jjas = np.zeros((118, len(lat), len(lon))) # 1900 to 2017 total 118 years

num_y = 0
for yyyy in range(1900, 2017):
    f0_single = f0.sel(time=f0.time.dt.year.isin([yyyy]))
    
    f0_single_jja  = f0_single.sel(time=f0_single.time.dt.month.isin([6, 7, 8,]))
    f0_single_jjas = f0_single.sel(time=f0_single.time.dt.month.isin([6, 7, 8, 9]))

    #print(f0_single_jja)
    pr_jja[num_y]  = np.nanmean(f0_single_jja['precip'].data, axis=0)
    pr_jjas[num_y] = np.nanmean(f0_single_jjas['precip'].data, axis=0)

    num_y += 1
    print(f"Success the year {yyyy}")

# Write to ncfile
ncfile  =  xr.Dataset(
{
    "PRECT_JJA":     (["time", "lat", "lon"], pr_jja),
    "PRECT_JJAS":     (["time", "lat", "lon"], pr_jjas),
},
coords={
    "time": (["time"], np.linspace(1900, 2017, 118)),
    "lat":  (["lat"],  lat),
    "lon":  (["lon"],  lon),
},
)

ncfile['PRECT_JJA'].attrs = f0_single.precip.attrs

ncfile.attrs['description'] = "Created on Huaibei Server cal_ERL_figs2_udel_jja_precip_1901_1955_linear_trend_240921.py uoe-code. 2024-9-21."

ncfile.to_netcdf("/home/sun/data/download_data/data/analysis_data/UDEL_JJA_JJAS_precip_1900_2017.nc")

