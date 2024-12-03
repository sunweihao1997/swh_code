'''
2024-4-2
This script is to convert the hourly OLR data into monthly OLR data through CDO
'''

import os

path_in = '/home/sun/data/other_data/down_ERA5_hourly_OLR_convert_float/'
path_out= '/home/sun/data/download/ERA5_OLR_monthly/'

data_list = os.listdir(path_in) ; data_list.sort()

for ff in data_list:
    os.system('cdo monmean ' + path_in + ff + ' ' + path_out + ff.replace('hourly', 'monthly'))