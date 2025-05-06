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

data_path = "/home/sun/data/download_data/data/chinabox/"

cesm_all     = xr.open_dataset(data_path + "china_new_ts_all.nc")
cesm_all_std = xr.open_dataset(data_path + "china_new_ts_allstd.nc") 
cesm_feu     = xr.open_dataset(data_path + "china_new_ts_fEU.nc")
cesm_feu_std = xr.open_dataset(data_path + "china_new_ts_fEUstd.nc") 
cru          = xr.open_dataset(data_path + "china_new_ts_cru.nc")
gpcc         = xr.open_dataset(data_path + "china_new_ts_gpcc.nc")

value_cesm_all    = cesm_all["pamod"].data
value_cesm_allstd = cesm_all_std["pastd"].data
value_cesm_feu    = cesm_feu["pf"].data
value_cesm_feustd = cesm_feu_std["pfstd"].data
value_cru         = cru["pcrumodsm"].data
value_gpcc        = gpcc["pgpccmodsm"].data

#print(value_cesm_all.shape)

colors = ['#8E72B5', '#D75455', '#333333', '#6BAE6C', '#F4A261']

# add linear fit
x = np.linspace(1901, 1955, 55)

y_gpcc = cal_moving_average(cal_moving_average(cal_moving_average(value_gpcc,  5), 5), 5)
y_allf = cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all,  5), 5), 5)
y_eu   = cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all - value_cesm_feu,  5), 5), 5)

A = np.vstack([x, np.ones(len(x))]).T  # 构造设计矩阵
m_gpcc, c_gpcc = np.linalg.lstsq(A, y_gpcc[:55], rcond=None)[0]  # 计算最小二乘法解
gpcc_fit = m_gpcc * x + c_gpcc

m_all, c_all = np.linalg.lstsq(A, y_allf[:55], rcond=None)[0]  # 计算最小二乘法解
all_fit  = m_all  * x + c_all

m_eu,  c_eu = np.linalg.lstsq(A, y_eu[:55], rcond=None)[0]  # 计算最小二乘法解
eu_fit   = m_eu * x + c_eu

#sys.exit()
plt.figure(figsize=(20, 10))

plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cru,  5), 5), 5),         color=colors[0],            label='CRU'  , linewidth=3)
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_gpcc,  5), 5), 5),        color=colors[1],                label='GPCC' , linewidth=3)
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all,  5), 5), 5),    color=colors[2],                label='All'  , linewidth=3)
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_feu,  5), 5), 5),    color=colors[3],     label='FixEU', linewidth=3)
#plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all - value_cesm_feu,  5), 5), 5),  markevery=2,  color='r',  marker='*',   label='All - FixEU')
plt.plot(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all - value_cesm_feu,  5), 5), 5),  color=colors[4], label='All - FixEU', linewidth=3)

plt.plot(np.linspace(1901, 1955, 55), gpcc_fit,  color=colors[1], linestyle='--', linewidth=3)
plt.plot(np.linspace(1901, 1955, 55), all_fit,  color=colors[2], linestyle='--', linewidth=3)
plt.plot(np.linspace(1901, 1955, 55), eu_fit,  color=colors[4], linestyle='--', linewidth=3)


#plt.xlim((1901, 1975))

plt.ylim((-0.75, 0.75))
plt.xticks(np.linspace(1900, 1990,  10))
plt.xlim((1900, 1975))

plt.xticks(fontsize=30, )  # x轴刻度
plt.yticks(fontsize=30, )  # y轴刻度


plt.fill_between(np.linspace(1901, 2000, 100), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all - value_cesm_allstd,  5), 5), 5), cal_moving_average(cal_moving_average(cal_moving_average(value_cesm_all + value_cesm_allstd,  5), 5), 5), color='lightgrey', alpha=0.7)
#plt.fill_between(np.linspace(1901, 2000, 100), cal_moving_average(feu,11)-cal_moving_average(feustd,11), cal_moving_average(feu,11)+cal_moving_average(feustd,11), color='darkgrey', alpha=0.7)

plt.legend(loc='lower right', fontsize=30)
plt.savefig('/home/sun/paint/ERL/ERL_figs3_ts_new_smooth.pdf', dpi=450)