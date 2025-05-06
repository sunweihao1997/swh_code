'''
2024-7-4
This script is to plot the time series of the experiments
'''
import numpy as np
import xarray as xr
import sys

def cal_moving_average(x, w):
        return np.convolve(x, np.ones(w), "same") / w

cru   = np.fromfile('/home/sun/data/download_data/data/timeseries/india_new_ts_cru.dat',    dtype=np.float32)
gpcc  = np.fromfile('/home/sun/data/download_data/data/timeseries/india_new_ts_gpcc.dat',   dtype=np.float32)
ctlall= np.fromfile('/home/sun/data/download_data/data/timeseries/india_new_ts_all.dat',    dtype=np.float32)
feu   = np.fromfile('/home/sun/data/download_data/data/timeseries/india_new_ts_fEU.dat',    dtype=np.float32)
ctlstd= np.fromfile('/home/sun/data/download_data/data/timeseries/india_new_ts_allstd.dat', dtype=np.float32)
feustd= np.fromfile('/home/sun/data/download_data/data/timeseries/india_new_ts_fEUstd.dat', dtype=np.float32)

import matplotlib.pyplot as plt

#print(ctlall)
#print((cru[10:-10] - )
#plt.plot(cal_moving_average(cru, 11), color='green')
#plt.plot(cal_moving_average(gpcc, 11), color='k')
#plt.plot(cal_moving_average(ctlall, 11), color='b')
#plt.plot(cal_moving_average(feu, 11), color='r')

print(len(cal_moving_average(ctlall,11)))

slope, intercept = np.polyfit(np.linspace(1901, 1955, 55), cal_moving_average(ctlall,11)[:55], 1)

# 输出斜率
print(f"斜率: {slope}")
#sys.exit()
plt.figure(figsize=(20, 14))
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cru   ,11), color='brown', label='CRU')
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(gpcc  ,11), color='k',     label='GPCC')
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(ctlall,11), color='b',     label='All')
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(feu   ,11), color='r',     label='FixEU')

plt.ylim((-0.5, 0.5))
plt.xlim((1910, 1990))
plt.xticks(np.linspace(1910, 1990, 9))

plt.tick_params(axis='both', labelsize=25)

plt.fill_between(np.linspace(1901, 2000, 100), cal_moving_average(ctlall,11)-cal_moving_average(ctlstd,11), cal_moving_average(ctlall,11)+cal_moving_average(ctlstd,11), color='lightgrey', alpha=0.7)
plt.fill_between(np.linspace(1901, 2000, 100), cal_moving_average(feu,11)-cal_moving_average(feustd,11), cal_moving_average(feu,11)+cal_moving_average(feustd,11), color='darkgrey', alpha=0.7)

plt.legend(fontsize=25)

plt.savefig('/home/sun/paint/ERL/ERL_fig1c_time_series_pr.pdf')

#file_path = 'your_file.dat'
#
#j = 0
#with open('/exports/csce/datastore/geos/users/s2618078/data/timeseries/india_new_ts_cru.dat', 'rb') as file:
#    for line in file.read():
##        print(line)
#        print(j)
#        j+=1