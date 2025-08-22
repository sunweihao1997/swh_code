'''
2025-8-18
This script is used to test reading the ship observation.
In the csv file, some latitude and longitude values are missing.
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

test_file = pd.read_csv(path_ship + "202506_3E7246.csv", encoding='gb2312')
print(test_file.head())