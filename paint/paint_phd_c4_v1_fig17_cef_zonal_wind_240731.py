'''
240731
This script is to calculate the time series of the CEF and zonal wind over Southern Bay of Bengal

Serves as fig in chapter C4, in order to show the relationship between CEF and zonal wind
'''
import xarray as xr
import numpy as np

# Read files
f1 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/climatic_daily_ERA5/single/10m_u_component_of_wind_climatic_daily.nc")
f2 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/climatic_daily_ERA5/single/10m_v_component_of_wind_climatic_daily.nc")

# Calculation
# 1. BOBCEF
f2_bobcef = f2.sel(lat=slice(5, -5), lon=slice(70, 90))
f2_somcef = f2.sel(lat=slice(5, -5), lon=slice(50, 60))
f1_zonal  = f1.sel(lat=slice(10, 0), lon=slice(80, 100))
#print(f1_bobcef['u10'].data)

bobcef    = np.average(np.average(f2_bobcef['v10'].data, axis=1), axis=1)
somcef    = np.average(np.average(f2_somcef['v10'].data, axis=1), axis=1)
uwind     = np.average(np.average(f1_zonal['u10'].data,  axis=1), axis=1)
#print(bobcef.shape)

# Plot
import matplotlib.pyplot as plt
import sys

fig, ax1 = plt.subplots(figsize=(10, 6))

# 折线图
ax1.plot(np.linspace(59, 120, 120 - 59 + 1), bobcef[59:121], label='BOB-CEF', color='deepskyblue', linewidth=2.25)
#sys.exit()
#ax1.plot(np.linspace(59, 120, 120 - 59 + 1), somcef[59:121], label='Somali-CEF', color='red')

ax1.set_xlabel('Date')

#ax1.legend(loc='upper left')

# 柱状图

ax1.bar(np.linspace(59, 120, 120 - 59 + 1), uwind[59:121], label='Zonal-Wind', color='dimgrey')
#ax2.set_ylabel('unit')
ax1.legend(loc='upper left')

ax1.set_xticks(np.array([60, 70, 80, 90, 100, 110, 120]))
ax1.set_xticklabels(["1-March", "10-March", "20-March", "1-April", "10-April", "20-April", "1-May"])

#plt.title('Line and Bar Chart on the Same Figure')
plt.savefig('/home/sun/paint/phd/phd_c4_fig2_BOBCEF_Zonalwind.pdf')