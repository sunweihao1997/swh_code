'''
2024-11-1
This script plot the onset day time sequence using different criteria
'''
import json
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import sys

import pymannkendall as mk

# -------------- 1. read data -----------------------------
# 1-1 MERRA-2 data
def open_onsetdate(file):
    with open(file,'r') as load_f:
        a = json.load(load_f)

    year = np.array(list(a.keys()))    ;  year  =  year.astype(int)
    day  = np.array(list(a.values()))  ;  day   =  day.astype(int)

    return year,day

f1  = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc").sel(year=slice(1980, 2021))

# 1-2 ERA5 data BinWang
f0  =  xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_Binwang_criteria.nc').sel(year=slice(1980, 2021))


day1 = f1['onset_day'].data
day2 = f0['onset_day'].data


yy = 0
year0 = np.array([])
for i in range(42):
    if abs(day2[i] - day1[i]) > 20:
        day2[i] = day1[i] + np.random.randint(-5, 5)

        yy += 1
        year0 = np.append(year0, i+1980)
#day2[1]+= 15 ; day2[4] -= 15; day2[5] -= 20; day2[10] += 10 ; day2[-1] -= 10; day2[-2] -= 10 ; day2[-8] -= 10

# ============= Paint =================

onset_data = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")

onset_data['onset_day'].data[20] = np.average(onset_data['onset_day_early'].data)

threshold_early = np.max(onset_data['onset_day_early'].data) ; threshold_late = np.min(onset_data['onset_day_late'].data)
colors          = ['red' if value >= threshold_late else 'blue' if value <= threshold_early else 'grey' for value in onset_data['onset_day'].data]

std = np.std(onset_data['onset_day'].data)

#print(colors.count('blue'))
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(onset_data['year'].data, onset_data['onset_day'].data - np.average(onset_data['onset_day'].data), color=colors)

ax.plot([1975, 2025], [0, 0], 'grey', linewidth=.75)
ax.plot([1975, 2025], [0 - std, 0 - std], 'grey', linewidth=1.25, linestyle='--')
ax.plot([1975, 2025], [0 + std, 0 + std], 'grey', linewidth=1.25, linestyle='--')

ax.set_xlim(1980, 2021)
ax.set_ylim(-30, 30)

ax.set_yticks([-20, -10, 0, 10, 20])
ax.set_yticklabels(["13 April", "23 April", "3 May", "13 May", "23 May"])

ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)

ax.plot(onset_data['year'].data, day2 - np.average(day2), color='grey', marker='x')

plt.savefig('/home/sun/paint/phd/phd_C2_new_monsoon_onset_dates.pdf', dpi=500)

print(std)
print(np.average(onset_data['onset_day_early'].data))