'''
2025-9-8
This script is to calculate statistics of wave rank using ERA5 data.

Note:
1. Wave Heght is instantaneous, not hourly mean.
'''
import xarray as xr
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

data_path = "/mnt/f/ERA5_wave/"
varname_list = ['swh', 'shts', 'shww'] # Significant height of combined wind waves and swell, swell component and wind wave component
hazard_name  = ['red', 'orange', 'yellow', 'blue'] # Hazard levels
hazard_value_nearland = [6, 4.5, 3.5, 2.5] 
hazard_value_offshore = [14, 9, 6, 4] 

test_file = "ERA5_wave.0.5x0.5.201104.nc"

def classify_day_offfshore(max_swh):
    '''
    Classify the day based on the maximum significant wave height.
    '''
    if max_swh >= hazard_value_offshore[0]:
        return 1  # Red
    elif max_swh >= hazard_value_offshore[1]:
        return 2  # Orange
    elif max_swh >= hazard_value_offshore[2]:
        return 3  # Yellow
    elif max_swh >= hazard_value_offshore[3]:
        return 4  # Blue
    else:
        return 'none'  # No hazard

def calculate_hazard_days(ncfile):
    '''
    Calculate the statistics of wave rank for a given month offshore.
    '''
    # 1. Group by days
    ncfile_grouped = ncfile.resample(valid_time='1D').max()

    print(ncfile_grouped)


if __name__ == "__main__":
    # Test
    ncfile = xr.open_dataset(os.path.join(data_path, test_file))
    calculate_hazard_days(ncfile)