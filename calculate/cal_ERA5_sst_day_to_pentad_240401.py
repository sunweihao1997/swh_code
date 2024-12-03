'''
2024-4-1 
This script is to convert daily data into pentad average for the ERA5 SST
'''
import xarray as xr
import numpy as np
import os

data_path = '/home/sun/data/ERA5_SST/day/'
data_list = os.listdir(data_path) ; data_list.sort()

ref_file  = xr.open_dataset(data_path + 'ERA5_SST_day_1976.nc')
lat       = ref_file.latitude.data ; lon      = ref_file.longitude.data

# Claim an array to save the pentad averaged data into one file
pentad_array = np.zeros((63, 73, len(lat), len(lon)))

# Claim an array to save the monthly averaged data into one file
month_array   = np.zeros((63, 12, len(lat), len(lon)))

# list includes amount of days per month
months_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Function for the pentad calculation
def pentad_cal(array0):
    pentad_array0 = np.zeros((73, len(lat), len(lon)))

    for i in range(73):
        pentad_array0[i] = np.average(array0[i*5 : i*5+5], axis=0)

    return pentad_array0

#print(np.sum(months_days[1:2]))
def monthly_cal(array0):
    month_array0 = np.zeros((12, len(lat), len(lon)))

    current_index = 0
    for i in range(12):
        start0 = np.sum(months_days[0:i])
        end0   = np.sum(months_days[0:i + 1])
        
        #print(f'It is in month {i+1}, start is {start0}, end is {end0}, length is {end0 - start0}')
        month_array0[i] = np.average(array0[int(start0):int(end0)], axis=0)

    return month_array0

# Start calculation
for nn in range(63):
    f0 = xr.open_dataset(data_path + data_list[nn])

    pentad_array[nn] = pentad_cal(f0.sst.data)

    month_array[nn]  = monthly_cal(f0.sst.data)

# Save them to the netcdf file
ncfile  =  xr.Dataset(
    {
        "pentad_sst": (["year", "pentad", "lat", "lon"], pentad_array[:, :, ::-1, :]),
        "month_sst":  (["year", "month",  "lat", "lon"], month_array[:, :, ::-1, :]),
    },
    coords={
        "year":  (["year"],  np.linspace(1959, 2021, 63)),
        "month": (["month"], np.linspace(1, 12, 12)),
        "pentad":(["pentad"],np.linspace(1, 73, 73)),
        "lat":   (["lat"],   lat[::-1]),
        "lon":   (["lon"],   lon),
    },
    )

ncfile["pentad_sst"].attrs  =  ref_file.sst.attrs
ncfile["month_sst"].attrs   =  ref_file.sst.attrs

ncfile.attrs["description"]  =  "Created on 2024-4-1 at ubuntu(Beijing), this file generated from cal_ERA5_sst_day_to_pentad_240401.py and is the pentad-averaged and monthly averaged from the ERA5 daily value."
ncfile.to_netcdf("/home/sun/data/ERA5_SST/pentad/ERA5_pentad_month_SST_1959-2021.nc")