'''
2025-3-18
This script is to plot the scatter between Q1 and OLR index
'''
import numpy as np
import xarray as xr

file_q = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_Maritime_Continent_monthly_March_April_diabatic_heating_statistical_quantity.nc")

file_olr1 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_mar.nc")
file_olr2 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_apr.nc")
file_olr3 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_feb.nc")
onset_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")

# calculate the March-April
olr_ma = (0.5 * (file_olr1['OLR_maritime'].data + file_olr2['OLR_maritime'].data))
olr_ma = (olr_ma - np.average(olr_ma)) / np.std(olr_ma)

q1 = (file_q['q1_avg'].data - np.average(file_q['q1_avg'].data))/np.std(file_q['q1_avg'].data)

import matplotlib.pyplot as plt

olr_ma_early = np.array([])
q1_early     = np.array([])
olr_ma_late  = np.array([])
q1_late      = np.array([])
for i in range(42):
    if i + 1980 in onset_file['year_early'].data:
        print(f"it is year {i+1980}, early year, xaxis is {olr_ma[i]}, onset day is {onset_file['onset_day'].data[i]}")
        olr_ma_early = np.append(olr_ma_early, olr_ma[i])
        q1_early = np.append(q1_early, q1[i])

    if i + 1980 in onset_file['year_late'].data:
        print(f"it is year {i+1980}, late year, xaxis is {olr_ma[i]}, onset day is {onset_file['onset_day'].data[i]}")
        olr_ma_late = np.append(olr_ma_late, olr_ma[i])
        q1_late = np.append(q1_late, q1[i])

plt.figure(figsize=(10, 10))
plt.scatter(olr_ma, q1[:42], s=100, color='grey', label='Group 1', alpha=0.8, edgecolors='grey')
plt.scatter(olr_ma_early, q1_early, s=100,color='blue', label='Group 1', alpha=1, edgecolors='blue')
plt.scatter(olr_ma_late, q1_late, s=100,color='red', label='Group 1', alpha=1, edgecolors='red')

plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

plt.savefig(('/home/sun/paint/lunwen/anomoly_analysis/v2_fig_scatter_OLR_heating.pdf'))