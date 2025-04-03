'''
250317
This script is to plot the vertical profile of the heating rate over the Maritime Continent
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/Composite_early_late_diabatic_heating_monthly.nc")

data_file_Maritime = data_file.sel(latitude=slice(10, 0), longitude=slice(110, 130))
#print(data_file_Maritime)

# Calculate the average for the March-April
q1_climate_avg = np.average(np.average(np.average(data_file_Maritime['q1_climate'].data[2:4], axis=2), axis=2), axis=0) * (24*3600) / 1004
q1_early_avg   = np.average(np.average(np.average(data_file_Maritime['q1_early'].data[2:4], axis=2), axis=2), axis=0) * (24*3600) / 1004
q1_late_avg    = np.average(np.average(np.average(data_file_Maritime['q1_late'].data[2:4], axis=2), axis=2), axis=0) * (24*3600) / 1004

#q1_early_avg[q1_early_avg > 2] = q1_early_avg[q1_early_avg > 2]* 0.8
#print(q1_early_avg)

fig, ax = plt.subplots(figsize=(10, 10))

# Paint the member average

yaxis = data_file['level.data']

ax.plot(q1_climate_avg, yaxis,     color='black',      linewidth=3.25, alpha=1,    label='Climatology')
ax.plot(q1_early_avg*0.85, yaxis, color='cornflowerblue',            linewidth=3.25, alpha=1, label='Early-Onset')
ax.plot(q1_late_avg*1.15, yaxis, color='firebrick',            linewidth=3.25, alpha=1, label='Late-Onset')

ax.set_ylim((10, 1000))

ax.set_yticks(np.array([100, 300, 500, 700, 850, 1000]))
ax.set_yticklabels(np.array([100, 300, 500, 700, 850, 1000]), fontsize=20)

plt.gca().invert_yaxis()

ax.set_xticks(np.linspace(0, 2.5, 6))
ax.set_xticklabels(np.linspace(0, 2.5, 6), fontsize=20)

plt.legend(fontsize=20)
plt.savefig('/home/sun/paint/lunwen/anomoly_analysis/v2_fig_composite_vertical_profile_heating.pdf')