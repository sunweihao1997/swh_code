'''
2024-7-12
This script is to calculate the climate variables for Maritime continent experiment

Note:
I only select the files in the last 20 years, to see whether it will be better
'''
import os
import numpy as np
import xarray as xr
import pickle
import matplotlib.pyplot as plt

exp_label = ['b1850_tx_maritime_o1_220730', 'b1850_tx_maritime_o2_220807']

# ============ File Information ============
path_in   = "/home/sun/segate/model_data/b1850_maritime/atmosphere/"

# ------------------ This process has been done and saved -----------------
#file_list = os.listdir(path_in)

#exp1      = []
#exp2      = []
#
#for ff in file_list:
#    if exp_label[0] in ff and "cam.h1" in ff:
#        exp1.append(ff)
#    elif exp_label[1] in ff and "cam.h1" in ff:
#        exp2.append(ff)
#    else:
#        continue
#
## 1. Save the result into file
## 将列表存储到文件
#with open('/home/sun/data/b1850_maritime_h1_cam_list_member1.pkl', 'wb') as file:
#    pickle.dump(exp1, file)
#
#with open('/home/sun/data/b1850_maritime_h1_cam_list_member2.pkl', 'wb') as file:
#    pickle.dump(exp2, file)
# -------------------------------------------------------------------------------Done

# 2. Load the saved file
with open('/home/sun/data/b1850_maritime_h1_cam_list_member1.pkl', 'rb') as file:
    exp1 = pickle.load(file) # 42 years * 365 + 1

with open('/home/sun/data/b1850_maritime_h1_cam_list_member2.pkl', 'rb') as file:
    exp2 = pickle.load(file) # only 10 years

del exp1
exp1 = os.listdir("/home/sun/segate/model_data/b1850_maritime/atmosphere/")

exp1.sort() ; exp2.sort()
# ========== End ===========

# ========== 2. Calculation Part ==========
def cal_climate_avg(list0, start, year, varname):
    # 2.1 claim the array0
    mtg_avg  = np.zeros((365))

    for i in range(start, start+year):
        start_loc = i*365
        print(f'Now it is year {i}, the first file is {list0[start_loc]}')
        for j in range(365):
            file_5  = xr.open_dataset(path_in + list0[start_loc + j]).sel(lev=slice(500, 200), lon=slice(80, 100), lat=slice(4,6))
            
            file_15 = xr.open_dataset(path_in + list0[start_loc + j]).sel(lev=slice(500, 200), lon=slice(80, 100), lat=slice(15,16))
            #print(file_15)

            #print((np.average(file_15['T'].data) - np.average(file_5['T'].data)))
            mtg_avg[j] += (np.average(file_15['T'].data) - np.average(file_5['T'].data)) / year

    return mtg_avg

def main():
    mtg_2040y = cal_climate_avg(exp1, 42, 9, 'T')

    plt.plot(mtg_2040y[90:150], marker='o')

    plt.savefig('test_mtg4.png')
    
if __name__ == '__main__':
    main()

