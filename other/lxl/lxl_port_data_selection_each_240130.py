'''
2024-1-30
This script is to select out the 15 stations' data for each port using different datasets, Because the single file includes total 323 stations

for the GFS data:
    
2023022200 path the t2m is missing

for the SD3 data
2023010412 path the hourly_tp is missing
2023021300/tp_hourly.nc is missing
'''
import numpy as np
import xarray as xr
import pandas as pd
import os
import math

# ==== Functions ====
def hanzi_to_pinyin(text):
    from pypinyin import pinyin, Style

    pinyin_list = pinyin(text, style=Style.NORMAL)

    # 将拼音列表转换为字符串
    pinyin_str = ''.join(word[0] for word in pinyin_list)
    
    return pinyin_str

def create_path_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        #print(f"路径 '{path}' 已创建。")
    else:
        print(f"路径 '{path}' 已经存在。")

def post_process_data(model_name, port_name, excel_port):

    path1 = datasets_path + model_name + '/'

    list1 = os.listdir(path1) ; list1.sort()

    ids   = excel_port['id'].values

    missing_file = [] ; a = 0

    ids_encode = []
    for iiii in ids:
        str_number = str(iiii)
        ids_encode.append(str_number.encode('UTF-8'))

    for date_name in list1:
        path2 = path1 + date_name + '/'
        
        if model_name == 'CMA':
            var_list = CMA_vars
        else:
            var_list = Mod_vars

        # Deal with each datafile under the path2

        # 1. Create the corresponding path under the target path
        out_path0 = out_path + port_name + '/' + model_name + '/' + date_name + '/'

        create_path_if_not_exists(out_path0)

        for vvvv in var_list:

            if os.path.isfile(path2 + vvvv):
                a += 1
            else:
                print(f"文件不存在: {path2 + vvvv}")
                missing_file.append(path2 + vvvv)

                continue

            f0 = xr.open_dataset(path2 + vvvv)

            # distribute to the path associated with port
            f0_filter = selection_station(f0, ids_encode)

            f0_filter.to_netcdf(out_path0 + vvvv)

        



def selection_station(ncfile, ids_encode):
    # distribute to the path associated with port
    num_station = ncfile['num_station'].data
    station_name= ncfile['station_name'].data

    location_list = []
    for iiii in ids_encode:
        for jjjj in range(len(station_name)):
            if iiii == station_name[jjjj]:
                #print('Has found {}'.format(iiii))
                location_list.append(jjjj)
            else:
                continue
            
    f0_filter  =  ncfile.isel(num_station=location_list)

    return f0_filter




def main():
    # ==== Read files ====

    path0    = '/home/sun/data/liuxl_sailing/other_data/'
    out_path = '/home/sun/data/liuxl_sailing/post_process/'  # The path which save the data for each datasets
    excel_path = '/home/sun/data/liuxl_sailing/other_data/'

    port_station_list = os.listdir(path0) ; port_station_list.sort()
    port_station_list_pinyin = []

    #print(len(port_station_list))
    #a     = port_station_list[1]
    #b     = a.split(" ")
    #c     = hanzi_to_pinyin(b[3])
    #print(c)

    # ==== process the data ====
    # 1. Create 45 paths associated with port name following out_path
    for port_name in port_station_list:
        name_split = port_name.split(" ")

        pinyin_convert = hanzi_to_pinyin(name_split[3])

        create_path_if_not_exists(out_path + pinyin_convert)

        port_station_list_pinyin.append(port_name)

    # 2. Deal with each dataset and save the result in each port path under /home/sun/data/liuxl_sailing/post_process/

    datasets_name = ['CMA', 'IFS', 'GFS', 'SD3']
    #datasets_name = ['SD3']
    datasets_path = '/home/sun/data/liuxl_sailing/'

    # The variables under the datasets are different
    CMA_vars = ['t2m.nc', 'tp_hourly.nc', 'ws10m.nc']
    Mod_vars = ['t2m.nc', 'tp_hourly.nc', 'u10m.nc', 'v10m.nc'] # GFS IFS SD3 is the same

    # The Structure of the model data is Model/dates/variables
    for nnnn in datasets_name:
        for i in range(len(port_station_list)):
            name_split2 = port_station_list[i].split(" ")

            print(f'Now it is dealing with {nnnn} for the port {hanzi_to_pinyin(name_split2[3])}, corresponding to excel {name_split2[3]}')

            excel0 = pd.read_excel(excel_path + port_station_list[i])
            post_process_data(nnnn, hanzi_to_pinyin(name_split2[3]), excel0)

if __name__ == '__main__':
    main()