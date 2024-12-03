'''
2024-1-25
This script is to evaluate the wind-speed and its distance to the harbour for the nearst stations
data is from CMA

The period is 2023-7-27 to 2023-7-28, according to the typhoon 杜苏芮, which enter into the Fujian Province
'''
import numpy as np
import xarray as xr
import pandas as pd
import os

# 1. Firstly, read the excel to get the 15 nearst stations to the Shanghai Harbour
#list0 = os.listdir('/data5/2019swh/liuxl_sailing/other_data/')
#print(list0)
data_path  = '/home/sun/data/liuxl_sailing/other_data/'
excel_name = 'The_nearst_ten_stations_to_This station to 厦门港 distance(km).xlsx'

df = pd.read_excel(data_path + excel_name)

ids = df['id'].values # This is the 15 nearst stations id

# 2. Second, read the CMA data

# ref_file is used to get location information
#ref_file = xr.open_dataset('/data5/2019swh/liuxl_sailing/CMA/2023012500/ws10m.nc')

#print(ref_file.station_lat.data)

file_path = '/home/sun/data/liuxl_sailing/CMA/'
time_path = '2023072612'

f0        = xr.open_dataset(file_path + time_path + '/' + 'ws10m.nc')

# 2.1 filter out the 15 stations we want

# Note: Please be cautious that the "station_name" in the file is not the same with the common trpe of the string, this type contains a sequence of octets (0-255)
# Detailed explaination can be found in https://stackoverflow.com/questions/6269765/what-does-the-b-character-do-in-front-of-a-string-literal
# For the data process, wo can employ .encode('UTF-8') to transfer it into common string type

ids_encode = []
for iiii in ids:
    str_number = str(iiii)
    ids_encode.append(str_number.encode('UTF-8'))

#print(ids_encode)
#f0.sel(num_station=ids_encode)

num_station = f0['num_station'].data
station_name= f0['station_name'].data
ws10m       = f0['ws10m'].data

location_list = []
for iiii in ids_encode:
    for jjjj in range(len(station_name)):
        if iiii == station_name[jjjj]:
            #print('Has found {}'.format(iiii))
            location_list.append(jjjj)
        else:
            continue

f0_filter  =  f0.isel(num_station=location_list)

# ----------------------------- At this time, the nearest 15 stations have been selected and the shape of variable in f0_filter is (240, 15) ---------------------------------------------

# 3. Calculation
# 3.1 daily evolution of the 15 stations, 24 hours

#print(f0_filter.time.data)
daily_ws = np.zeros((48, 15))
for i in range(15):
    daily_ws[:, i] = f0_filter['ws10m'].data[:48, i]

# 3.2 Plot the time-evolution
import matplotlib.pyplot as plt

time_str = []
for i in range(48):
    str_t = str(f0['time'].data[i])
    time_str.append(str_t[7:9])

fig, axs = plt.subplots(figsize=(30, 10))
axs.plot(np.linspace(0, 47, 48), daily_ws)

axs.set_xticks(range(0, 47, 3))

#axs.set_xlabel(str_t)
plt.savefig('/home/sun/paint/liuxl_sail/CMA_0727_0728_wd10m.png')

