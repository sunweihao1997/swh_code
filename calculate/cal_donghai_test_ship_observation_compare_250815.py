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

# --- Detect the encoding of the first file ---
#file_path = path_ship + file_list[0]
#with open(file_path, 'rb') as f:
#    result = chardet.detect(f.read(10000))  # 检测前 1 万字节
#print(result)



test_file = pd.read_csv(path_ship + file_list[0], encoding='gb2312')
#print(test_file.columns)

def compose_time(year, month, day, hour):
    """
    组合时间字符串
    """
    dt = datetime.strptime(f"{year}-{month:02d}-{day:02d} {hour:02d}:00", "%Y-%m-%d %H:%M")

    utc = pytz.UTC
    bj = pytz.timezone("Asia/Shanghai")

    dt_utc = utc.localize(dt)  # 标注成 UTC 时间
    dt_bj = dt_utc.astimezone(bj)  # 转成北京时间

    return dt_bj

for index,row in test_file.iterrows():
    year  = row['年']
    month = row['月']
    day = row['日']
    hour = row['时']

#    print(year, month, day, hour)

#    print(type(year))
    
    time_str = compose_time(int(year), int(month), int(day), int(hour))
    
    # 打印时间字符串
    print(time_str)
    
