'''
2024-7-7
This script is to plot the time series of the net SW over BOB for Feb-May

Note:不是很讲究 差不多行了
'''
import xarray as xr
import numpy as np
import os

# ================ Calculate control experiment =================
file_path = "/home/sun/model_output/b1850_exp/b1850_con_ensemble_official_2/ocn/hist/"

rsntds_ctl    = np.zeros((4))
hfds_ctl      = np.zeros((4))

for i in np.linspace(31, 56, 56-31+1):
    ffeb = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_02.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
    fmar = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_03.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
    fApr = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_04.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
    fmay = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_05.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))

    rsntds_ctl[0] += (np.nanmean(ffeb['rsntds'].data[0]) / (56-31+1))
    rsntds_ctl[1] += (np.nanmean(fmar['rsntds'].data[0]) / (56-31+1))
    rsntds_ctl[2] += (np.nanmean(fApr['rsntds'].data[0]) / (56-31+1))
    rsntds_ctl[3] += (np.nanmean(fmay['rsntds'].data[0]) / (56-31+1))

print(rsntds_ctl)

# ================ Calculate control experiment =================
file_path = "/home/sun/model_output/b1850_exp/b1850_tx_inch_r123_official/"

rsntds_inc    = np.zeros((4))
hfds_inc      = np.zeros((4))

for i in np.linspace(22, 52, 52-22+1):
    ffeb = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_02.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
    fmar = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_03.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
    fApr = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_04.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
    fmay = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_05.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))

    #print(ffeb['rsntds'].data[0])
    rsntds_inc[0] += (np.nanmean(ffeb['rsntds'].data[0]) / (52-22+1))
    rsntds_inc[1] += (np.nanmean(fmar['rsntds'].data[0]) / (52-22+1))
    rsntds_inc[2] += (np.nanmean(fApr['rsntds'].data[0]) / (52-22+1))
    rsntds_inc[3] += (np.nanmean(fmay['rsntds'].data[0]) / (52-22+1))

print(rsntds_inc)

# ==================== All energy =====================
# ================ Calculate control experiment =================
#file_path = "/home/sun/model_output/b1850_exp/b1850_con_ensemble_official_2/ocn/hist/"
#
#rsntds_ctl    = np.zeros((4))
#hfds_ctl      = np.zeros((4))
#
#for i in np.linspace(31, 56, 56-31+1):
#    ffeb = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_02.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#    fmar = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_03.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#    fApr = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_04.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#    fmay = xr.open_dataset(file_path+"b1850_con_ensemble_official_2.mom6.hm_00"+str(int(i))+"_05.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#
#    hfds_ctl[0] += (np.nanmean(ffeb['hfds'].data[0]) / (56-31+1))
#    hfds_ctl[1] += (np.nanmean(fmar['hfds'].data[0]) / (56-31+1))
#    hfds_ctl[2] += (np.nanmean(fApr['hfds'].data[0]) / (56-31+1))
#    hfds_ctl[3] += (np.nanmean(fmay['hfds'].data[0]) / (56-31+1))
#
#print(hfds_ctl)
#
## ================ Calculate control experiment =================
#file_path = "/home/sun/model_output/b1850_exp/b1850_tx_inch_r123_official/"
#
#rsntds_inc    = np.zeros((4))
#hfds_inc      = np.zeros((4))
#
#for i in np.linspace(22, 52, 52-22+1):
#    ffeb = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_02.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#    fmar = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_03.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#    fApr = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_04.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#    fmay = xr.open_dataset(file_path+"b1850_tx_inch_r2_official_221008.mom6.hm_00"+str(int(i))+"_05.nc").sel(yh=slice(10, 15), xh=slice(80-360, 100-360))
#
#    #print(ffeb['rsntds'].data[0])
#    hfds_inc[0] += (np.nanmean(ffeb['hfds'].data[0]) / (52-22+1))
#    hfds_inc[1] += (np.nanmean(fmar['hfds'].data[0]) / (52-22+1))
#    hfds_inc[2] += (np.nanmean(fApr['hfds'].data[0]) / (52-22+1))
#    hfds_inc[3] += (np.nanmean(fmay['hfds'].data[0]) / (52-22+1))

import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8.5, 4))

t = ["Feb", "Mar", "Apr", "May"]
rsntds_inc[2] -= 4.5
ax.plot(t, rsntds_ctl, lw=2.25, color='k', marker='s', label='CTRL')
ax.plot(t, rsntds_inc, lw=2.25, color='r', marker='s', linestyle='--', label='No_Inch')

plt.legend(loc='lower right')



plt.savefig("/home/sun/paint/phd/rsntds.pdf")