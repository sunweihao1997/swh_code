'''
2024-11-4
This script is to plot the monsoon onset dates
'''
import xarray as xr
import numpy as np
import os
import sys
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
from matplotlib.path import Path
import matplotlib.patches as patches

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
ax.set_ylim(-25, 25)

ax.set_yticks([-20, -10, 0, 10, 20])
ax.set_yticklabels(["13 April", "23 April", "3 May", "13 May", "23 May"])

ax.tick_params(axis='x', labelsize=13)
ax.tick_params(axis='y', labelsize=13)



plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/ERA5_MTG_index_monsoon_onset_time_series_1980_2021.pdf', dpi=500)

print(np.average(onset_data['onset_day'].data))