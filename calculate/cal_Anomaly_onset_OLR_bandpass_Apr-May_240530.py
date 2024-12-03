'''
2024-5-30
This script is to use the bandpass filtered OLR to calculate the early/late years composite
'''
import xarray as xr
import numpy as np
import sys

sys.path.append("/home/sun/mycode_copy/calculate/")
from cal_Anomaly_onset_OLR_evolution_240515 import screen_early_late

# data about onset early/late years
onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years.nc")
onset_day_file_42 = onset_day_file.sel(year=slice(1980, 2021)) #42 years

# data about the bandpass OLR data
OLR_file       = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_OLR_bandpass_filter_30_80.nc")

lat = OLR_file.lat.data ; lon = OLR_file.lon.data
# test the timeaxis
#time_year      = np.unique(OLR_file.time.dt.year)
#print(time_year.shape)

# ============== Part of calculate the composite average ====================

early_years, late_years = screen_early_late(onset_day_file_42['onset_day'].data)

# claim the array for the result
composite_early = np.zeros((40, len(lat), len(lon))) # 40 means -30 0 9 
composite_late  = composite_early.copy()

# Start calculating
for ee in early_years:
    # 1. select the one year
    OLR_file_1year = OLR_file.sel(time=OLR_file.time.dt.year.isin([ee]))

    # 2. re-arrange and select the 40 time day
    onset_day_1year = onset_day_file_42.sel(year=ee)

    olr_rearrange   = OLR_file_1year['olr'].data[int(onset_day_1year.onset_day.data-1-30):int(onset_day_1year.onset_day.data-1-30+40)]

    # 3. add to the array
    composite_early += (olr_rearrange / len(early_years))

for ee in late_years:
    # 1. select the one year
    OLR_file_1year = OLR_file.sel(time=OLR_file.time.dt.year.isin([ee]))

    # 2. re-arrange and select the 40 time day
    onset_day_1year = onset_day_file_42.sel(year=ee)

    olr_rearrange   = OLR_file_1year['olr'].data[int(onset_day_1year.onset_day.data-1-30):int(onset_day_1year.onset_day.data-1-30+40)]

    # 3. add to the array
    composite_late += (olr_rearrange / len(late_years))

# Save to the ncfile
ncfile  =  xr.Dataset(
{
    "olr_early": (["time", "lat", "lon"], composite_early),
    "olr_late":  (["time", "lat", "lon"], composite_late),
},
coords={
    "time": (["time"], np.linspace(-30, 9, 40)),
    "lat":  (["lat"],  OLR_file.lat.data),
    "lon":  (["lon"],  OLR_file.lon.data),
},
)
ncfile['olr_early'].attrs['units'] = 'W m**-2'
ncfile['olr_late'].attrs['units'] = 'W m**-2'

ncfile.attrs['description']  =  'Created on 2024-5-30 by cal_Anomaly_onset_OLR_bandpass_composite_240530.py. This is the composite average of the onset early/late year for OLR, note that the result is 8-80 bandpass result'
ncfile.to_netcdf("/home/sun/data/composite/early_late_composite/ERA5_OLR_bandpass_3080_early_late_composite.nc")