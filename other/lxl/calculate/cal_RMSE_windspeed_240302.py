'''
2024-3-2
This script is to  visualize the RMSE for the GFS, IFS, Tianji, during a process of typhoon
'''
import os
import xarray as xr
import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error


# ============================ excel reading ===========================================
# Read the information about the 15 nearest stations to the Quanzhou gang

xlsx_path = '/home/sun/data/liuxl_sailing/other_data/'
xlsx_name = 'The_nearst_ten_stations_to_This station to 泉州港 distance(km).xlsx'

df        = pd.read_excel(xlsx_path + xlsx_name)
ids       = df['id'].values # ids: the 15 nearest station's id. Note that the ids have been sorted by the distances to the port. 
                            #      Thus you can use selection to decide how many stations are used to calculate the spatial average RMSE
                            #      for example: ids = ids[:3] means only the nearest stations were used to calculate the RMSE 
ids       = ids[:3]         #      Here we use 3 stations

# ============================ Read the observation data (CMA) =========================
# choose which datasets to read and date time

def filter_out_data(fcma, id_num):
    '''
        This function is to select the nearest stations from the 15 stations
    '''
    ids_encode = []
    for ii in id_num:
        str_number = str(ii)
        ids_encode.append(str_number.encode('UTF-8'))

    station_name= fcma['station_name'].data

    location_list = []
    for ii in ids_encode:
        for jjjj in range(len(station_name)):
            if ii == station_name[jjjj]:
                location_list.append(jjjj)

    fcma_select    = fcma.isel(num_station=location_list)

    return fcma_select

model_name = 'CMA'
date_name  = '2023072612'

file_path  = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + model_name + '/' + date_name + '/'

file_name  = 'ws10m.nc'

fcma       = xr.open_dataset(file_path + file_name)

fcma       = filter_out_data(fcma, ids)

# =================== end of the <Read the observation data (CMA)> ================================

# =================== Read the model prediction data (GFS, EC, Tianji) ============================
# ------------------- U10M ------------------------------------------------------------------------
model_name = 'GFS' # GFS, IFS, SD3
date_name  = '2023072612'

file_path  = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + model_name + '/' + date_name + '/'

file_name  = 'u10m.nc'

fgfs_u     = xr.open_dataset(file_path + file_name)

fgfs_u     = filter_out_data(fgfs_u, ids)

model_name = 'IFS' # GFS, IFS, SD3
date_name  = '2023072612'

file_path  = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + model_name + '/' + date_name + '/'

file_name  = 'u10m.nc'

fifs_u     = xr.open_dataset(file_path + file_name)

fifs_u     = filter_out_data(fifs_u, ids)

model_name = 'SD3' # GFS, IFS, SD3
date_name  = '2023072612'

file_path  = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + model_name + '/' + date_name + '/'

file_name  = 'u10m.nc'

ftij_u     = xr.open_dataset(file_path + file_name)

ftij_u     = filter_out_data(ftij_u, ids)

# ------------------- V10M ------------------------------------------------------------------------
model_name = 'GFS' # GFS, IFS, SD3
date_name  = '2023072612'

file_path  = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + model_name + '/' + date_name + '/'

file_name  = 'v10m.nc'

fgfs_v     = xr.open_dataset(file_path + file_name)

fgfs_v     = filter_out_data(fgfs_v, ids)

model_name = 'IFS' # GFS, IFS, SD3
date_name  = '2023072612'

file_path  = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + model_name + '/' + date_name + '/'

file_name  = 'v10m.nc'

fifs_v     = xr.open_dataset(file_path + file_name)

fifs_v     = filter_out_data(fifs_v, ids)

model_name = 'SD3' # GFS, IFS, SD3
date_name  = '2023072612'

file_path  = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + model_name + '/' + date_name + '/'

file_name  = 'v10m.nc'

ftij_v     = xr.open_dataset(file_path + file_name)

ftij_v     = filter_out_data(ftij_v, ids)

# =================== end of the <Read the prediction data> ================================

# =================== calculation of the wind speed =====================

ws_gfs     = np.average(np.sqrt(np.square(fgfs_u['u10m'].data) + np.square(fgfs_v['v10m'].data)), axis=1)
ws_ifs     = np.average(np.sqrt(np.square(fifs_u['u10m'].data) + np.square(fifs_v['v10m'].data)), axis=1)
ws_tij     = np.average(np.sqrt(np.square(ftij_u['u10m'].data) + np.square(ftij_v['v10m'].data)), axis=1)
# Above are the wind speed, 3 station average, 240 hours start from date_time

ws_cma     = np.average(fcma['ws10m'].data, axis=1) 

# =================== end of the <calculation of the wind speed> ============================

# =================== Calculation of the RMSE =============================
# Reference: https://stackoverflow.com/questions/17197492/is-there-a-library-function-for-root-mean-square-error-rmse-in-python

from math import sqrt

# The 48 hours from the start prediction

rms_gfs  =  np.array([])
rms_ifs  =  np.array([])
rms_tij  =  np.array([])

for i in range(48):
    rms_gfs = np.append(rms_gfs, sqrt(mean_squared_error([np.average(ws_cma[i])], [np.average(ws_gfs[i])])))
    rms_ifs = np.append(rms_ifs, sqrt(mean_squared_error([np.average(ws_cma[i])], [np.average(ws_ifs[i])])))
    rms_tij = np.append(rms_tij, sqrt(mean_squared_error([np.average(ws_cma[i])], [np.average(ws_tij[i])])))

import matplotlib.pyplot as plt

fig, axs = plt.subplots(figsize=(30, 18))
axs.plot(np.linspace(0, 47, 48), rms_gfs, color='black', label='GFS', linewidth=3, marker='o', markersize=12)
axs.plot(np.linspace(0, 47, 48), rms_ifs, color='royalblue', label='IFS', linewidth=3, marker='o', markersize=12)
axs.plot(np.linspace(0, 47, 48), rms_tij, color='red', label='Tianji', linewidth=3, marker='o', markersize=12)



plt.legend(fontsize=30)

axs.set_xticks(range(0, 47, 3))
axs.set_yticks(range(0, 13, 2))

axs.set_ylabel('RMSE', fontsize=40)

time_axis = ["7-26:12-00", "7-26:15-00", "7-26:18-00", "7-26:21-00", "7-27:00-00", "7-27:03-00", "7-27:06-00", "7-27:09-00", 
                "7-12:00-00", "7-27:15-00", "7-27:18-00", "7-27:21-00", "7-28:00-00", "7-28:03-00", "7-28:06-00", "7-28:09-00"]
axs.set_xticklabels(time_axis, fontsize=30, rotation=45)
axs.set_yticklabels(range(0, 13, 2), fontsize=35)

axs.set_title('Chongwu (23.5km), Nanan (33.2km), Xiuyu (48.8km)', loc='right', fontsize=40)
#axs.set_title('Chongwu (23.5km)', loc='right', fontsize=40)

#axs.set_xlabel(str_t)
plt.savefig('/home/sun/paint/liuxl_sail/model_evaluation_RMSE_3station.png')