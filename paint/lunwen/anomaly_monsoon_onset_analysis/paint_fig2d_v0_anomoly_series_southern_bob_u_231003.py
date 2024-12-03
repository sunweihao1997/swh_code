'''
2023-10-03
This script paint the fig2c in the paper, which is the series of the uwind anomoly in Spring

Different with previous picture, here replace the 10m wind by 925 hPa wind
'''
import xarray as xr
import numpy as np
import os

# ------------ 1. Information --------------------------------
# 1.1 Range
lat_range = slice(10, 0)
lon_range = slice(80, 90)
level     = 925

# 1.2 file path; Metion: These files are saved in daily file
path0 = '/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5/u_component_of_wind/'
path1 = '/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/u_component_of_wind/'

# 1.3 Get the file list in three types
list0 = os.listdir(path0) ; list0.sort()
list1 = os.listdir(path1) ; list1.sort()

climate_series = os.listdir(path0) ; climate_series.sort()
early_series   = [] ; late_series = []
for ffff in list1:
    if 'early' in ffff:
        early_series.append(ffff)
    elif 'late' in ffff:
        late_series.append(ffff)

#print(len(early_series))
#print(len(late_series))
early_anomoly = np.array([]) ; late_anomoly = np.array([])
for dddd in range(365):
    f0 = xr.open_dataset(path0 + climate_series[dddd]).sel(lat=lat_range, lon=lon_range, lev=925)
    f1 = xr.open_dataset(path1 + early_series[dddd]).sel(lat=lat_range, lon=lon_range, lev=925)
    f2 = xr.open_dataset(path1 + late_series[dddd]).sel(lat=lat_range, lon=lon_range, lev=925)

    early_anomoly = np.append(early_anomoly, np.average(f1['u'].data) - np.average(f0['u'].data))
    late_anomoly  = np.append(late_anomoly,  np.average(f2['u'].data) - np.average(f0['u'].data))





# --------------- 3. Paint the Picture --------------------------
def paint_early_late_cef_intensity(early_cef, late_cef):
    import matplotlib.pyplot as plt

    # 3.1 Tick Settings
    # time range from 20 Feb to 1 Jun
    x_tick = [50, 59, 68, 78, 90, 99, 109, 120, 129, 139, 151]
    x_label = ['20 Feb', '1 Mar', '10 Mar', '20 Mar', '1 Apr', '10 Apr', '20 Apr', '1 May', '10 May', '20 May', '1 Jun']

    # y-axis
    y_tick = np.linspace(-4.5, 4.5, 19)

    # 3.2 Plot the early year using blue color
    plt.bar(np.linspace(1, 365, 365, dtype=int), early_cef, color='blue')

    # 3.3 Plot the late year using red color
    plt.bar(np.linspace(1, 365, 365, dtype=int), late_cef, color='red')

    # 3.4 Set limitation
    plt.ylim(-4.8, 4.8)
    plt.xlim(45, 155)

    # 3.5 Tick information
    plt.xticks(x_tick, x_label, rotation=45)
    plt.yticks(y_tick)

    plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/abnormal_southbob_u_series.pdf')

def main():
    paint_early_late_cef_intensity(early_anomoly, late_anomoly)


if __name__ == '__main__':
    main()