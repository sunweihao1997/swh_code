'''
2025-2-27
This script is to decompose the MTG-series into different temporal scale (higher than 20) (20-70) (lower than 70)
'''
import xarray as xr
import numpy as np
import os
import sys
from scipy.signal import butter, filtfilt

#  1. Read the File
f0_mtg = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_daily_MTG_series_BOB_1940_2022.nc")

# =============== Filtering Functions ===================
def highpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)  # 双向滤波避免相移
    return filtered_data


def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    filtered_data = filtfilt(b, a, data)  # 双向滤波避免相移
    return filtered_data

def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)  # 双向滤波避免相移
    return filtered_data

# ============= 1. Calculate the filtered series ================
sampling_frequency = 1  # 逐日采样，Fs=1
# 1.1 High-Pass filter 20-days
cutoff_frequency = 1 / 11  # 20 天周期的倒数

filtered_signal_highpass = np.zeros(f0_mtg["MTG_series"].data.shape)
for i in range(83):
    filtered_signal_highpass[i, :] = highpass_filter(f0_mtg["MTG_series"].data[i], cutoff_frequency, sampling_frequency)

# 1.2 Band-Pass filter 2070-days
low_cutoff = 1 / 80  # 70 天周期的倒数
high_cutoff = 1 / 12  # 20 天周期的倒数
sampling_frequency = 1  # 逐日采样，Fs=1

filtered_signal_bandpass = np.zeros(f0_mtg["MTG_series"].data.shape)
for i in range(83):
    filtered_signal_bandpass[i, :] = bandpass_filter(f0_mtg["MTG_series"].data[i], low_cutoff, high_cutoff, sampling_frequency)

# 1.3 Low-Pass filter 2070-days
low_cutoff = 1 / 81

filtered_signal_lowpass = np.zeros(f0_mtg["MTG_series"].data.shape)
for i in range(83):
    filtered_signal_lowpass[i, :] = lowpass_filter(f0_mtg["MTG_series"].data[i], low_cutoff, sampling_frequency)


# ============== 2. Short test: Select one year to show the filtered data ==============
# import matplotlib.pyplot as plt
# 
# # 创建数据
# x = np.linspace(1, 365, 365)
# 
# # 创建绘图
# plt.figure(figsize=(8, 6))
# plt.plot(x, filtered_signal_highpass[70], label='High', color='r', linestyle='-')  # 红色实线
# plt.plot(x, filtered_signal_bandpass[70], label='Band', color='g', linestyle='--') # 绿色虚线
# plt.plot(x, filtered_signal_lowpass[70],  label='Low',  color='b', linestyle=':')  # 蓝色点线
# 
# # 添加标题和标签
# 
# # 显示图形
# plt.savefig("test.png")
# ============== 2. The end of the test part =============================================

# ============== Add coordinate information to the result ================================
# ----------- save to the ncfile ------------------
ncfile  =  xr.Dataset(
{
    "highpass_MTG": (["year", "day"], filtered_signal_highpass),
    "bandpass_MTG": (["year", "day"], filtered_signal_bandpass),
    "lowpass_MTG":  (["year", "day"], filtered_signal_lowpass),
    "original_MTG":  (["year", "day"], f0_mtg["MTG_series"].data),
},
coords={
    "year": (["year"], f0_mtg["year"].data),
    "day":  (["day"],  f0_mtg["day"].data),
},
)

# ============= 3. Allocate the early and late onset years ===============
onset_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")
ncfile_all = ncfile.sel(year=onset_file.year.data)
ncfile_ear = ncfile.sel(year=onset_file.year_early.data)
ncfile_lat = ncfile.sel(year=onset_file.year_late.data)

# ============ 4. Plot the picture =======================================
# Here I plot all the three situations: early, normal and late
import matplotlib.pyplot as plt

# 4.1 Early onset situation
fig, ax = plt.subplots(figsize=(10, 8))

date_axi = np.linspace(1, 365, 365)

avg_time   = int(np.average(onset_file['onset_day_early'])) ; interval = 25 ; print(f"The average of the early onset is {avg_time}")
start_time = avg_time - interval ; end_time = avg_time + interval
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_ear['lowpass_MTG'].data, axis=0)[start_time:end_time],  label='71-day lowpass filter', color='blue', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_ear['bandpass_MTG'].data, axis=0)[start_time:end_time], label='21-70-day bandpass filter', color='red', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_ear['highpass_MTG'].data, axis=0)[start_time:end_time], label='20-day highpass filter', color='green', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_ear['original_MTG'].data, axis=0)[start_time:end_time], label='20-day highpass filter', color='black', lw=2)

ax.plot([-1*interval, 1*interval], [0, 0], 'k--', lw=0.75)

ax.fill_between(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_ear['bandpass_MTG'].data, axis=0)[start_time:end_time] - 0.75*np.std(ncfile_ear['bandpass_MTG'].data, axis=0)[start_time:end_time], np.average(ncfile_ear['bandpass_MTG'].data, axis=0)[start_time:end_time] + 0.75*np.std(ncfile_ear['bandpass_MTG'].data, axis=0)[start_time:end_time], alpha=0.2, color='red')

ax.set_ylim((-1.5, 1.5))
ax.set_xlim((-1*interval+3, 1*interval-3))

ax.legend(loc='upper left')

#ax.set_xticks([90, 100, 110, 120, 130, 140, 150])
ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
#ax.set_xticklabels(["1-April", "10-April", "20-April", "1-May", "10-May", "20-May", "31-May"])
ax.tick_params(axis='x', labelsize=17.5, labelcolor='k')
ax.tick_params(axis='y', labelsize=17.5, labelcolor='k')


plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/Article_Anomaly_ISO_v2_fig1_decomposed_mtg_early_onset.pdf')

# 4.2 Normal onset situation
fig, ax = plt.subplots(figsize=(10, 8))

date_axi = np.linspace(1, 365, 365)

avg_time   = int(np.average(onset_file['onset_day'])) ; interval = 25 ; print(f"The average of the normal onset is {avg_time}")
start_time = avg_time - interval - 3 ; end_time = avg_time + interval - 3
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_all['lowpass_MTG'].data, axis=0)[start_time:end_time],  label='71-day lowpass filter', color='blue', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_all['bandpass_MTG'].data, axis=0)[start_time:end_time], label='21-70-day bandpass filter', color='red', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_all['highpass_MTG'].data, axis=0)[start_time:end_time], label='20-day highpass filter', color='green', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_all['original_MTG'].data, axis=0)[start_time:end_time], label='20-day highpass filter', color='black', lw=2)

ax.plot([-1*interval, 1*interval], [0, 0], 'k--', lw=0.75)

ax.fill_between(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_all['bandpass_MTG'].data, axis=0)[start_time:end_time] - 0.75*np.std(ncfile_all['bandpass_MTG'].data, axis=0)[start_time:end_time], np.average(ncfile_all['bandpass_MTG'].data, axis=0)[start_time:end_time] + 0.75*np.std(ncfile_all['bandpass_MTG'].data, axis=0)[start_time:end_time], alpha=0.2, color='red')

ax.set_ylim((-1.5, 1.5))
ax.set_xlim((-1*interval+3, 1*interval-3))

ax.legend(loc='upper left')

#ax.set_xticks([90, 100, 110, 120, 130, 140, 150])
ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
#ax.set_xticklabels(["1-April", "10-April", "20-April", "1-May", "10-May", "20-May", "31-May"])
ax.tick_params(axis='x', labelsize=17.5, labelcolor='k')
ax.tick_params(axis='y', labelsize=17.5, labelcolor='k')


plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/Article_Anomaly_ISO_v2_fig1_decomposed_mtg_normal_onset.pdf')

# 4.3 Late onset situation
fig, ax = plt.subplots(figsize=(10, 8))

date_axi = np.linspace(1, 365, 365)

avg_time   = int(np.average(onset_file['onset_day_late'])) ; interval = 25 ; print(f"The average of the late onset is {avg_time}")
start_time = avg_time - interval - 3 ; end_time = avg_time + interval -3
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_lat['lowpass_MTG'].data, axis=0)[start_time:end_time],  label='71-day lowpass filter', color='blue', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_lat['bandpass_MTG'].data, axis=0)[start_time:end_time], label='21-70-day bandpass filter', color='red', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_lat['highpass_MTG'].data, axis=0)[start_time:end_time], label='20-day highpass filter', color='green', lw=2)
ax.plot(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_lat['original_MTG'].data, axis=0)[start_time:end_time], label='20-day highpass filter', color='black', lw=2)

ax.plot([-1*interval, 1*interval], [0, 0], 'k--', lw=0.75)

ax.fill_between(np.linspace(-1*interval, 1*interval, 2*interval), np.average(ncfile_lat['bandpass_MTG'].data, axis=0)[start_time:end_time] - 0.75*np.std(ncfile_lat['bandpass_MTG'].data, axis=0)[start_time:end_time], np.average(ncfile_lat['bandpass_MTG'].data, axis=0)[start_time:end_time] + 0.75*np.std(ncfile_lat['bandpass_MTG'].data, axis=0)[start_time:end_time], alpha=0.2, color='red')

ax.set_ylim((-1.5, 1.5))
ax.set_xlim((-1*interval+3, 1*interval-3))

ax.legend(loc='upper left')

#ax.set_xticks([90, 100, 110, 120, 130, 140, 150])
ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
#ax.set_xticklabels(["1-April", "10-April", "20-April", "1-May", "10-May", "20-May", "31-May"])
ax.tick_params(axis='x', labelsize=17.5, labelcolor='k')
ax.tick_params(axis='y', labelsize=17.5, labelcolor='k')


plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/Article_Anomaly_ISO_v2_fig1_decomposed_mtg_late_onset.pdf')

# Added function: calculating the contribution 
print(np.average(ncfile_lat['original_MTG'].data, axis=0)[avg_time-5-3:avg_time+5-3]) # 0.67
print(np.average(ncfile_lat['highpass_MTG'].data, axis=0)[avg_time-5-3:avg_time+5-3]) # 0.08
print(np.average(ncfile_lat['bandpass_MTG'].data, axis=0)[avg_time-5-3:avg_time+5-3]) # 0.35
print(np.average(ncfile_lat['lowpass_MTG'].data, axis=0)[avg_time-5-3:avg_time+5-3])  # 0.24
