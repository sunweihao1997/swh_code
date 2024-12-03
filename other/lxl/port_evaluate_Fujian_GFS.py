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
import math

# 1. Firstly, read the excel to get the 15 nearst stations to the Shanghai Harbour
#list0 = os.listdir('/data5/2019swh/liuxl_sailing/other_data/')
#print(list0)
data_path  = '/home/sun/data/liuxl_sailing/other_data/'
excel_name = 'The_nearst_ten_stations_to_This station to 厦门港 distance(km).xlsx'

df = pd.read_excel(data_path + excel_name)

ids = df['id'].values # This is the 15 nearst stations id
#print(ids)

# 2. Second, read the CMA data

# ref_file is used to get location information
#ref_file = xr.open_dataset('/data5/2019swh/liuxl_sailing/GFS/20230125/ws10m.nc')

#print(ref_file.station_lat.data)

file_path = '/home/sun/data/liuxl_sailing/GFS/'
time_path = '2023072700'

f0_u        = xr.open_dataset(file_path + time_path + '/' + 'u10m.nc')
f0_v        = xr.open_dataset(file_path + time_path + '/' + 'v10m.nc')

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

num_station = f0_u['num_station'].data
station_name= f0_u['station_name'].data

location_list = []
for iiii in ids_encode:
    for jjjj in range(len(station_name)):
        if iiii == station_name[jjjj]:
            #print('Has found {}'.format(iiii))
            location_list.append(jjjj)
        else:
            continue

f0_ufilter  =  f0_u.isel(num_station=location_list)
f0_vfilter  =  f0_v.isel(num_station=location_list)

u10m       = f0_ufilter['u10m'].data
v10m       = f0_vfilter['v10m'].data

ws10m      = np.zeros(u10m.shape)

for i in range(len(f0_ufilter.time.data)):
    for j in range(15):
        ws10m[i, j] = math.sqrt(math.pow(u10m[i, j], 2) + math.pow(v10m[i, j], 2))


# ----------------------------- At this time, the nearest 15 stations have been selected and the shape of variable in f0_filter is (240, 15) ---------------------------------------------

# 3. Calculation
# 3.1 daily evolution of the 15 stations, 24 hours

#print(f0_filter.time.data)
ws10m_2day = ws10m[:48, :]

# 3.2 Plot the time-evolution
import matplotlib.pyplot as plt


fig, axs = plt.subplots(figsize=(30, 10))
axs.plot(np.linspace(0, 47, 48), ws10m_2day)

axs.set_xticks(range(0, 47, 3))

axs.set_title('GFS', loc='right', fontsize=20)
axs.set_title('Start at 07-20 00:00', loc='left', fontsize=20)

axs.set_ylim((0, 30))

#axs.set_xlabel(str_t)
plt.savefig('/home/sun/paint/liuxl_sail/Xiamen_harbour_DuSuRui_20230727_20230728_ws10m_timeseries_GFS.png')

