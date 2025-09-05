'''
2025-8-27
This script is used to combine the ship observation data and ERA5 data(by interpolation).

This is a complement of the original script, to deal with the negative values
'''
import numpy as np
import pandas as pd
import xarray as xr
import os
import chardet
from datetime import datetime
import pytz
import re
from zoneinfo import ZoneInfo

path_ship = "/mnt/f/ERA5_ship/ERA5_ship_addtime/"
#print(os.listdir(path_ship))

file_list = os.listdir(path_ship)
file_list.sort()

test_file = pd.read_csv(path_ship + "202506_3E7246.csv", encoding='gb2312')
#print(test_file.head())
#
#print(test_file.iloc[5]['年'])


def calc_rh(t2, td2):
    """
    t2  : 2米气温 (单位 K 或 °C)
    td2 : 2米露点温度 (单位 K 或 °C)
    返回: 相对湿度 RH (%)
    """
    # 如果是开尔文，转成摄氏
    if np.nanmedian(t2) > 200:
        t2 = t2 - 273.15
    if np.nanmedian(td2) > 200:
        td2 = td2 - 273.15

    # 饱和水汽压 (hPa)
    es = 6.112 * np.exp((17.67 * t2) / (t2 + 243.5))
    # 实际水汽压 (hPa)
    e = 6.112 * np.exp((17.67 * td2) / (td2 + 243.5))

    rh = (e / es) * 100
    return np.clip(rh, 0, 100)  # 限制在 0-100 %

# =========== Function to return the target file ============
def ERA5_interpolation(csv_file):
    '''
    This script use time to find the target file.
    '''
    ERA5_path = "/mnt/f/ERA5_ship/"
    vars_list = ['surface_pressure', 'mean_sea_level_pressure', '2m_temperature', '2m_dewpoint_temperature', '10m_v_component_of_wind', '10m_u_component_of_wind']

    #csv_file = csv_file.rename(columns=lambda x: re.sub(r'\(.*\)', '', x))
    csv_file = csv_file.drop('皮温', axis=1) # I do not know what PIWEN is, so I drop it.
    csv_file = csv_file.drop('能见度', axis=1) # I do not know what PIWEN is, so I drop it.

    # Results Values
    u10 = [] ; v10 = [] ; t2 = [] ; td2 = [] ; msl = [] ; sp = []

    for row in csv_file.itertuples():
        # 检查每一行是否含有 NaN
        if any(pd.isna(value) for value in row[1:]):  # row[1:] 是去掉索引部分的行数据
            u10.append(np.nan) ; v10.append(np.nan) ; t2.append(np.nan) ; td2.append(np.nan) ; msl.append(np.nan) ; sp.append(np.nan)
            continue
        else:
            year = row.年
            month = row.月
            day = row.日
            hour = row.时

            lat  = row.纬度
            lon  = row.经度

            if lon < 0:
                lon = lon + 360

            time_compose = datetime.strptime(f"{year}-{month:02d}-{day:02d} {hour:02d}:00", "%Y-%m-%d %H:%M")

            time_beijing = time_compose.replace(tzinfo=ZoneInfo("Asia/Shanghai"))

            time_utc = time_beijing.astimezone(ZoneInfo("UTC"))

            utc_year  = time_utc.year
            utc_month = time_utc.month

            #print(time_compose) ; print(time_utc)

            #print(time_compose)

            #print(f"Processing time: {year}-{month:02d}-{day:02d} {hour:02d}:00")
            # Generate the filenames
            file_name_10u = f"ERA5_hourly_single.0.5x0.5.10m_u_component_of_wind.{utc_year}{utc_month:02d}.nc"
            file_name_10v = f"ERA5_hourly_single.0.5x0.5.10m_v_component_of_wind.{utc_year}{utc_month:02d}.nc"
            file_name_2t  = f"ERA5_hourly_single.0.5x0.5.2m_temperature.{utc_year}{utc_month:02d}.nc"
            file_name_2d  = f"ERA5_hourly_single.0.5x0.5.2m_dewpoint_temperature.{utc_year}{utc_month:02d}.nc"
            file_name_msl = f"ERA5_hourly_single.0.5x0.5.mean_sea_level_pressure.{utc_year}{utc_month:02d}.nc"
            file_name_sp  = f"ERA5_hourly_single.0.5x0.5.surface_pressure.{utc_year}{utc_month:02d}.nc"
            #print(file_name_10u)

            file_name_list = [file_name_10u, file_name_10v, file_name_2t, file_name_2d, file_name_msl, file_name_sp]
            vars_list      = [u10, v10, t2, td2, msl, sp]
            vars_name_list = ['u10', 'v10', 't2m', 'd2m', 'msl', 'sp']
            for num in np.arange(len(file_name_list)):
                
                # Check the same time
                f_single_var = xr.open_dataset(os.path.join(ERA5_path, file_name_list[num])).sel(valid_time=time_utc.replace(tzinfo=None))
                

                f_single_var_interp = f_single_var.interp(latitude=lat, longitude=lon, method="linear")

                # Add the interpolated value to the list
                vars_list[num].append(f_single_var_interp[vars_name_list[num]].values.item())
                #print(ftest_interp)

                #print(f"Processing file: {file_name}")
            
    # Save the results to the csv file
    csv_file['ERA5_10m_u_component_of_wind'] = u10
    csv_file['ERA5_10m_v_component_of_wind'] = v10
    csv_file['ERA5_2m_temperature']          = t2
    csv_file['ERA5_2m_dewpoint_temperature'] = td2
    csv_file['ERA5_mean_sea_level_pressure'] = msl
    csv_file['ERA5_surface_pressure']        = sp
    csv_file['ERA5_wind_speed']              = (csv_file['ERA5_10m_u_component_of_wind']**2 + csv_file['ERA5_10m_v_component_of_wind']**2)**0.5
    csv_file['ERA5_wind_direction']          = (270 - np.degrees(np.arctan2(v10, u10))) % 360
    csv_file['ERA5_relative_humidity'] = calc_rh(csv_file['ERA5_2m_temperature'],csv_file['ERA5_2m_dewpoint_temperature'])

    return csv_file


outpath = "/mnt/f/ERA5_ship/add_ERA5_interpolation_beijing_time/"

for fff in file_list:
    print(f"Processing file: {fff}")
    if not fff.endswith('.csv'):
        continue
    
    input_file = pd.read_csv(path_ship + fff, encoding='gb2312')
    input_file = input_file.rename(columns=lambda x: re.sub(r'\(.*\)', '', x))
    #print(input_file.columns)
    
    # === 新增判断：如果经度列没有负值，跳过 ===
    if not (input_file['经度'] < 0).any():
        print(f"  -> 文件 {fff} 不含负经度，跳过插值。")
        continue

    output_file = ERA5_interpolation(input_file)
    output_file.to_csv(os.path.join(outpath, fff), index=False, encoding='gb2312')
    print(f"  -> 文件 {fff} 已完成插值并保存。")