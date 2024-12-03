'''
2024-4-1
This script is to cat the ERA5 SST data into one file per year
'''
from cdo import *
cdo = Cdo()

#import numpy as np
#import xarray as xr
import os
import sys

path_data = '/home/sun/wd_disk/era5_sst/'

# Get the file list
list_data = os.listdir(path_data) ; list_data.sort()

# Function: extract The data in each year
def get_one_years_data(year):
    tag0 = "temperature_" + str(year)

    one_year = []
    for ffff in list_data:
        if tag0 in ffff:
            one_year.append(ffff)
        else:
            continue
    
    one_year.sort()

    return one_year

# Start data process
out_path = '/home/sun/data/ERA5_SST/'

start_year = 1959 ; end_year = 2021
for yyyy in range(start_year, end_year + 1):
    oneyear_list = get_one_years_data(int(yyyy))

    if len(oneyear_list) != 12:
        sys.exit(f"The length of the {yyyy} is {len(oneyear_list)}")
    else:
        cdo.cat(input = [(path_data + x) for x in oneyear_list],output = out_path + "ERA5_SST_day_" + str(int(yyyy)) + ".nc", options = '-b F64')