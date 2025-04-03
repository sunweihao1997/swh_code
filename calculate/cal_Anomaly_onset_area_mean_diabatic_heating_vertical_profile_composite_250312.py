'''
20250312
This script use composite average diabatic heating to calculate the vertical profile of the area-mean Q1 over the Maritime Continent
'''

import xarray as xr
import numpy as np
import os
import sys

data_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/Composite_early_late_diabatic_heating_monthly.nc")

data_file_Maritime = data_file.sel(latitude=slice(10, 0), longitude=slice(110, 130)) #; print(data_file_Maritime)

area_mean_climate = 24 * 3600 * np.average(np.average(np.average(data_file_Maritime["q1_climate"].data[2:4], axis=2), axis=2), axis=0) / 1004
area_mean_early   = 3600 * np.average(np.average(np.average(data_file_Maritime["q1_early"].data[2:4], axis=2), axis=2), axis=0)
area_mean_late    = 3600 * np.average(np.average(np.average(data_file_Maritime["q1_late"].data[2:4], axis=2), axis=2), axis=0)

print(area_mean_climate)