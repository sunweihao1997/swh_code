'''
2024-6-2
This script is to use the bandpass filtered OLR to calculate the early/late years composite

and the April-May average

2024-8-16
new : According to Wu advice, I extend time-axis to -70 days in order to compare the late and early circulation
'''
import xarray as xr
import numpy as np
import sys
import pandas as pd

sys.path.append("/home/sun/mycode_copy/calculate/")
from cal_Anomaly_onset_OLR_evolution_240515 import screen_early_late

# data about onset early/late years
onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years.nc")
onset_day_file_42 = onset_day_file.sel(year=slice(1980, 2021)) #42 years

# data about the bandpass OLR data
OLR_file       = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_tp_bandpass_filter_30_80.nc")

lat = OLR_file.lat.data ; lon = OLR_file.lon.data
# test the timeaxis
#time_year      = np.unique(OLR_file.time.dt.year)
#print(time_year.shape)

# ============== Part of calculate the composite average ====================

early_years, late_years = screen_early_late(onset_day_file_42['onset_day'].data)

# claim the array for the result
composite_early = np.zeros((40, len(lat), len(lon))) # 40 means -30 0 9 
composite_late  = np.zeros((80, len(lat), len(lon))) # 80 means -70 0 9

aprmay_early    = np.zeros((61, len(lat), len(lon)))
aprmay_late     = np.zeros((61, len(lat), len(lon)))

# --------- Start calculating the composite average ----------
for ee in early_years:
    # 1. select the one year
    OLR_file_1year = OLR_file.sel(time=OLR_file.time.dt.year.isin([ee]))

    # 2. re-arrange and select the 40 time day
    onset_day_1year = onset_day_file_42.sel(year=ee)

    olr_rearrange   = OLR_file_1year['tp_filter'].data[int(onset_day_1year.onset_day.data-1-30):int(onset_day_1year.onset_day.data-1-30+40)]

    # 3. add to the array
    composite_early += (olr_rearrange / len(early_years))

for ee in late_years:
    # 1. select the one year
    OLR_file_1year = OLR_file.sel(time=OLR_file.time.dt.year.isin([ee]))

    # 2. re-arrange and select the 40 time day
    onset_day_1year = onset_day_file_42.sel(year=ee)

    olr_rearrange   = OLR_file_1year['tp_filter'].data[int(onset_day_1year.onset_day.data-1-30-40):int(onset_day_1year.onset_day.data-1-30-40+80)]

    # 3. add to the array
    composite_late += (olr_rearrange / len(late_years))

# ----------- Start calculating the Apr-May average -------------
# Start calculating the composite average
for ee in early_years:
    # 1. select the one year
    OLR_file_1year = OLR_file.sel(time=OLR_file.time.dt.year.isin([ee]))

    # 2. re-arrange and select the 40 time day
    onset_day_1year = onset_day_file_42.sel(year=ee)

    olr_rearrange   = OLR_file_1year['tp_filter'].data[90:90+61]

    # 3. add to the array
    aprmay_early += (olr_rearrange / len(early_years))

for ee in late_years:
    # 1. select the one year
    OLR_file_1year = OLR_file.sel(time=OLR_file.time.dt.year.isin([ee]))

    # 2. re-arrange and select the 40 time day
    onset_day_1year = onset_day_file_42.sel(year=ee)

    olr_rearrange   = OLR_file_1year['tp_filter'].data[90:90+61]

    # 3. add to the array
    aprmay_late += (olr_rearrange / len(late_years))

# Save to the ncfile
ncfile  =  xr.Dataset(
{
    "tp_early": (["time_composite", "lat", "lon"],        composite_early),
    "tp_late":  (["time_composite_extend", "lat", "lon"], composite_late),
#    "tp_am_early": (["time", "lat", "lon"], aprmay_early),
#    "tp_am_late":  (["time", "lat", "lon"], aprmay_late),
},
coords={
    "time_composite": (["time_composite"], np.linspace(-30, 9, 40)),
    "time_composite_extend": (["time_composite_extend"], np.linspace(-70, 9, 80)),
    "lat":  (["lat"],  OLR_file.lat.data),
    "lon":  (["lon"],  OLR_file.lon.data),
    "time": (["time"],  pd.date_range(start='2000-04-01', end='2000-05-31', freq='D'))
},
)
ncfile['tp_early'].attrs['units'] = 'mm day-1'
ncfile['tp_late'].attrs['units'] = 'mm day-1'

ncfile.attrs['description']  =  'Created on 2024-8-16 by cal_Anomaly_onset_precipitation_bandpass_composite_new_240816.py. This is the composite average of the onset early/late year for precipitation, note that the result is 30-80 bandpass result. New edition means that the time axis for late years expand to -70 days'
ncfile.to_netcdf("/home/sun/data/composite/early_late_composite/ERA5_precipitation_bandpass_3080_early_late_composite_time_extend.nc")