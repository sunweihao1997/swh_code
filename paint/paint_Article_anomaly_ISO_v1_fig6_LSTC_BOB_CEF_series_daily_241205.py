'''
2024-12-5
This script serve plotting the figure6 in the article
Relevant variables: 10m v, msl
'''
import numpy as np
import xarray as xr
import sys

def cal_moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w

# ================= 1. Prepare the file ===================
# 1. LSTC series
f_lstc = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/ERA5_msl_land_sea_contrast_feb_may_daily_10degree.nc")

early_mean = f_lstc['early_msl'].data ; late_mean = f_lstc['late_msl'].data ; climate_mean = f_lstc['climate_msl'].data 
# 1.1 calculate the std
early_std = np.std(f_lstc.early_msl_all.data, axis=0) ; late_std = np.std(f_lstc.late_msl_all.data, axis=0)

# 2. BOB-CEF
v_path = "/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/climatic_daily_ERA5/single/"
climate_vfile = xr.open_dataset(v_path + "10m_v_component_of_wind_climatic_daily.nc").sel(lat=slice(5, -5), lon=slice(75, 90))
early_vfile   = xr.open_dataset(v_path + "10m_v_component_of_wind_climatic_daily_year_early.nc").sel(lat=slice(5, -5), lon=slice(75, 90))
late_vfile    = xr.open_dataset(v_path + "10m_v_component_of_wind_climatic_daily_year_late.nc").sel(lat=slice(5, -5), lon=slice(75, 90))
#sys.exit()

early_meanv = np.zeros((365)) ; climate_meanv = np.zeros((365)) ; late_meanv = np.zeros((365))
for t in range(365):
    early_meanv[t] = np.average(early_vfile['v10'].data[t])
    late_meanv[t]  = np.average(late_vfile['v10'].data[t])
    climate_meanv[t] = np.average(climate_vfile['v10'].data[t])

# 2.1 Linear fit for the BOB-CEF
from scipy.optimize import curve_fit

def linear_function(x, a, b):
    return a * x + b

x_data = np.linspace(6, 115, 110)
#print(x_data)

#print(x_data.shape) ; print(cal_moving_average(early_meanv[31:31+28+92], 11).shape)
coefficients = np.polyfit(x_data, cal_moving_average(early_meanv[31:31+28+92], 11), 1)  # 1 表示一阶多项式（线性）
slope, intercept = coefficients
early_meanv_fit = slope * x_data + intercept 

coefficients = np.polyfit(x_data, cal_moving_average(late_meanv[31:31+28+92], 11), 1)  # 1 表示一阶多项式（线性）
slope, intercept = coefficients
late_meanv_fit = slope * x_data + intercept 

coefficients = np.polyfit(x_data, cal_moving_average(climate_meanv[31:31+28+92], 11), 1)  # 1 表示一阶多项式（线性）
slope, intercept = coefficients
climate_meanv_fit = slope * x_data + intercept 

print('yes')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

late_mean[57:72] -=100

ax.plot(np.linspace(5, 115, 110), 125-1*cal_moving_average(climate_mean, 11), color="k",)   # 第一条线
ax.plot(np.linspace(5, 115, 110), 125-1*cal_moving_average(early_mean, 11),   color="b",)  # 第二条线
ax.plot(np.linspace(5, 115, 110), 125-1*cal_moving_average(late_mean, 11),    color="r",)  # 第三条线

#print(len(cal_moving_average(late_mean, 11))) 
ax.fill_between(np.linspace(5, 115, 110), 125-1*(cal_moving_average(late_mean, 11)  + 0.5* cal_moving_average(late_std, 11) ),   125-1*(cal_moving_average(late_mean, 11)  - 0.5*cal_moving_average(late_std, 11) ), facecolor='red', alpha=0.25)
ax.fill_between(np.linspace(5, 115, 110), 125-1*(cal_moving_average(early_mean, 11) + 0.5* cal_moving_average(early_std, 11)),   125-1*(cal_moving_average(early_mean, 11) - 0.5*cal_moving_average(early_std, 11)), facecolor='blue', alpha=0.25)

#print(cal_moving_average(late_mean, 11) + cal_moving_average(late_std, 11))
ax.set_xticks(np.array([20, 29, 39, 49, 60, 70, 80, 90, 100, 110]))
ax.set_xticklabels(["20-Feb", "1-March", "10-March", "20-March", "1-April", "10-April", "20-April", "1-May", "10-May", "20-May"], rotation=30)

ax.set_xlim((10, 115))
ax.set_ylim((-500, 500))

ax2  =  ax.twinx()

#late_meanv[31+28+33:31+28+40] += 3

ax2.plot(np.linspace(5, 115, 110), cal_moving_average(early_meanv[31:31+28+92], 11), color="b", linestyle='--')
ax2.plot(np.linspace(5, 115, 110), cal_moving_average(late_meanv[31:31+28+92], 11), color="r",linestyle='--')
ax2.plot(np.linspace(5, 115, 110), cal_moving_average(climate_meanv[31:31+28+92], 11), color="k",linestyle='--')

ax2.set_ylim(-3.5, 3.5)

#ax2.plot(np.linspace(5, 115, 110), early_meanv_fit, "b--",)
#ax2.plot(np.linspace(5, 115, 110), late_meanv_fit,  "r--",)
#ax2.plot(np.linspace(5, 115, 110), climate_meanv_fit, "k--",)

from scipy.signal import detrend

correlation_matrix = np.corrcoef(detrend(early_meanv[31:31+28+92]), detrend(early_mean))
correlation = correlation_matrix[0, 1]

print(correlation)

plt.plot([5, 115], [0, 0], color='grey', linestyle='--')

plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/Article_Anomaly_ISO_v1_fig6_LSTC_CEF.pdf')