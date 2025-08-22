'''
2025-8-15
This script is used to test reading the ship observation
'''
import numpy as np
import pandas as pd
import xarray as xr
import os
import chardet
from datetime import datetime
import pytz

path_ship = "/mnt/f/ERA5_ship/ship_data/"
#print(os.listdir(path_ship))

file_list = os.listdir(path_ship)
file_list.sort()

test_file = pd.read_csv(path_ship + file_list[0], encoding='gb2312')
#print(test_file.columns)

def compose_time(year, month, day, hour):
    """
    组合时间字符串
    """
    dt = datetime.strptime(f"{year}-{month:02d}-{day:02d} {hour:02d}:00", "%Y-%m-%d %H:%M")

    utc = pytz.UTC
    bj = pytz.timezone("Asia/Shanghai")

    # 创建 UTC 时间
    dt_utc = datetime(year, month, day, hour, tzinfo=utc)
    # 转成北京时间
    dt_bj = dt_utc.astimezone(bj)

    return dt_utc.strftime("%Y-%m-%d %H:%M"), dt_bj.strftime("%Y-%m-%d %H:%M")

for ff in file_list:
    if not ff.endswith('.csv'):
        continue
    file_path = os.path.join(path_ship, ff)
    
    # 检查文件是否完整
    if not os.path.exists(file_path):
        print(f"文件不存在：{file_path}")
        continue
    
    # 读取文件
    try:
        file_single = pd.read_csv(file_path, encoding='gb2312')
        print(f"读取文件成功：{file_path}")
    except Exception as e:
        print(f"读取文件失败：{file_path}，错误信息：{e}")
        continue

    time_str_list_utc = []
    time_str_list_bj  = []

    for index,row in file_single.iterrows():
        year  = row['年']
        month = row['月']
        day = row['日']
        hour = row['时']


        time_str_utc, time_str_bj = compose_time(int(year), int(month), int(day), int(hour))

        time_str_list_utc.append(time_str_utc)
        time_str_list_bj.append(time_str_bj)

    file_single['time_utc'] = time_str_list_utc
    file_single['time_bj'] = time_str_list_bj

    file_single.to_csv("/mnt/f/ERA5_ship/ERA5_ship_addtime/" + ff, index=False, encoding='gb2312')
    
