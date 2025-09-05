'''
2025-8-22
calculate true wind from relative wind and ship movement

Task Link:https://www.notion.so/257d5b19b11d80ec9586d290c7415f73?source=copy_link
'''
import numpy as np
import pandas as pd
import xarray as xr
import os
import chardet
from datetime import datetime
import pytz
import re

path_data = "/mnt/f/ERA5_ship/add_ERA5_interpolation_beijing_time/"
list_file = os.listdir(path_data)

# exclude other data
for filename in list_file:
    if not filename.endswith('.csv'):
        list_file.remove(filename)

# =========== Function to return the target file ============
def cal_donghai_ship_converse_relative_wind2truewind(csv_file):

    true_wind = [] ; true_wind_direction = []

    for index, row in csv_file.iterrows():
        if any(pd.isna(value) for value in row[1:]):
            true_wind.append(np.nan) ; true_wind_direction.append(np.nan)
            continue
        
        else:
            direction_ship = row['航向'] # unit: 0-360 degree
            speed_ship = row['航速'] * 0.2777 # unit: km/h -> m/s
            direction_wind = row['风向']
            speed_wind = row['风速'] # unit: m/s

            rad_d_ship =   np.deg2rad(direction_ship)
            rad_d_wind =   np.deg2rad(direction_wind)

            # Calculate the true wind components
            u_true = speed_wind * np.sin(rad_d_wind) + speed_ship * np.sin(rad_d_ship)
            v_true = speed_wind * np.cos(rad_d_wind) + speed_ship * np.cos(rad_d_ship)

            # Calculate the true wind speed and direction
            true_wind_speed = np.sqrt(u_true**2 + v_true**2)
            true_wind_direction_rad = np.arctan2(u_true, v_true)
            true_wind_direction_deg = np.rad2deg(true_wind_direction_rad) % 360

            true_wind.append(true_wind_speed)
            true_wind_direction.append(true_wind_direction_deg)

    csv_file['真风速'] = true_wind
    csv_file['真风向'] = true_wind_direction

    return csv_file

for file in list_file:
    print(f"Processing file: {file}")
    df = pd.read_csv(path_data + file, encoding='gb2312')
    df_result = cal_donghai_ship_converse_relative_wind2truewind(df)
    df_result.to_csv("/mnt/f/ERA5_ship/donghai_ship_true_wind/" + file, index=False, encoding='utf-8-sig')
    print(f"Finished processing and saved: {file}")