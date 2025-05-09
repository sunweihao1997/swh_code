'''
2025-3-29
This script is to calculate the composite difference between the early and late onsets, for 200 hPa wind and 500 hPa Omega
'''
import xarray as xr
import numpy as np
import os
import sys


sys.path.append("/home/sun/mycode/module/")
from module_sun import *

# Read the file
data_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/monthly/ERA5_1980_2021_monthly_200hpa_UVZ_500hpa_w.nc")
#print(data_file)
onset_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")

# claim the array for early and late
early_u = np.zeros((6, 721, 1440)) ; late_u = np.zeros((9, 721, 1440))
early_v = np.zeros((6, 721, 1440)) ; late_v = np.zeros((9, 721, 1440))
early_w = np.zeros((6, 721, 1440)) ; late_w = np.zeros((9, 721, 1440))

num_early = 0 ; num_late = 0
for i in range(1980, 2023):
    if i in onset_file['year_early'].data:
        early_u[num_early] = 0.5 * (data_file['Mar_u'].data[i - 1980] + data_file['Apr_u'].data[i - 1980])
        early_v[num_early] = 0.5 * (data_file['Mar_v'].data[i - 1980] + data_file['Apr_v'].data[i - 1980])
        early_w[num_early] = 0.5 * (data_file['Mar_w'].data[i - 1980] + data_file['Apr_w'].data[i - 1980])

        num_early += 1

    elif i in onset_file['year_late'].data:
        late_u[num_early] = 0.5 * (data_file['Mar_u'].data[i - 1980] + data_file['Apr_u'].data[i - 1980])
        late_v[num_early] = 0.5 * (data_file['Mar_v'].data[i - 1980] + data_file['Apr_v'].data[i - 1980])
        late_w[num_early] = 0.5 * (data_file['Mar_w'].data[i - 1980] + data_file['Apr_w'].data[i - 1980])

        num_late += 1

# calculate the composite difference
diff_u = (np.average(early_u, axis=0) - np.average(late_u, axis=0))
diff_v = (np.average(early_v, axis=0) - np.average(late_v, axis=0))
diff_w = (np.average(early_w, axis=0) - np.average(late_w, axis=0))

# calculate the statistical test
from scipy import stats

u_test_p = np.zeros((721, 1440))
v_test_p = np.zeros((721, 1440))
w_test_p = np.zeros((721, 1440))

for i in range(721):
    for j in range(1440):
        t_stat, p_value = stats.ttest_ind(early_u[:, i, j], late_u[:, i, j], equal_var=False)
        u_test_p[i, j] = p_value

        t_stat, p_value = stats.ttest_ind(early_v[:, i, j], late_v[:, i, j], equal_var=False)
        v_test_p[i, j] = p_value

        t_stat, p_value = stats.ttest_ind(early_w[:, i, j], late_w[:, i, j], equal_var=False)
        w_test_p[i, j] = p_value

# Write to ncfile
ncfile  =  xr.Dataset(
{
    "diff_u": (["lat", "lon"], diff_u),
    "diff_v": (["lat", "lon"], diff_v),
    "diff_w": (["lat", "lon"], diff_w),
    "p_u":    (["lat", "lon"], u_test_p),
    "p_v":    (["lat", "lon"], v_test_p),
    "p_w":    (["lat", "lon"], w_test_p),
},
coords={
    "lat": (["lat"], data_file.lat.data),
    "lon": (["lon"], data_file.lon.data),
},
)
ncfile.attrs['description']  =  'This data file is generated by the script /home/sun/swh_code/calculate/cal_Anomaly_onset_composite_difference_early_late_for_200wind_500omega_250329.py.  This file is the composite difference between early and late onset, for u200, v200, and w500. The p is the student t test about the variable.'
ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/Composite_diffence_early_late_onset_years_u200_v200_w500.nc")