'''
2024-10-16
This script is to refine the fig1c, using the new data Massimo passed me
'''
import xarray as xr
import numpy as np
import sys
import os
import scipy
import matplotlib.pyplot as plt
#from cdo import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

def cal_moving_average(x, w):
        return np.convolve(x, np.ones(w), "same") / w

data_path = "/home/sun/data/download_data/data/renearfinaldraft/"

cesm_all     = xr.open_dataset(data_path + "india_new_ts_all.nc")
cesm_all_std = xr.open_dataset(data_path + "india_new_ts_allstd.nc") 
cesm_feu     = xr.open_dataset(data_path + "india_new_ts_fEU.nc")
cesm_feu_std = xr.open_dataset(data_path + "india_new_ts_fEUstd.nc") 
cru          = xr.open_dataset(data_path + "india_new_ts_cru.nc")
gpcc         = xr.open_dataset(data_path + "india_new_ts_gpcc.nc")

value_cesm_all    = cesm_all["pamod"].data
value_cesm_allstd = cesm_all_std["pastd"].data
value_cesm_feu    = cesm_feu["pf"].data
value_cesm_feustd = cesm_feu_std["pfstd"].data
value_cru         = cru["pcrumodsm"].data
value_gpcc        = gpcc["pgpccmodsm"].data

print(value_cesm_all.shape)

colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#D62728']


plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cru,  5), 5), 5),         color=colors[0],            label='CRU'  )
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_gpcc,  5), 5), 5),        color=colors[1],                label='GPCC' )
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all,  5), 5), 5),    color=colors[2],                label='All'  )
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_feu,  5), 5), 5),    color=colors[3],     label='FixEU')
#plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all - value_cesm_feu,  5), 5), 5),  markevery=2,  color='r',  marker='*',   label='All - FixEU')
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all - value_cesm_feu,  5), 5), 5),  color='r', label='All - FixEU')


#plt.xlim((1901, 1975))

plt.ylim((-0.5, 0.5))
plt.xticks(np.linspace(1900, 1990,  10))
plt.xlim((1900, 1975))
plt.fill_between(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all - value_cesm_allstd,  5), 5), 5), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all + value_cesm_allstd,  5), 5), 5), color='lightgrey', alpha=0.7)
#plt.fill_between(np.linspace(1901, 2000, 100), cal_moving_average(feu,11)-cal_moving_average(feustd,11), cal_moving_average(feu,11)+cal_moving_average(feustd,11), color='darkgrey', alpha=0.7)

plt.legend()
plt.savefig('ts_new_smooth.png', dpi=450)