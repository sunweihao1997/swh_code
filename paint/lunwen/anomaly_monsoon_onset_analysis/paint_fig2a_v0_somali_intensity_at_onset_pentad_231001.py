'''
20231001
This script is for the paper article Fig2a
This picture includes information: 40 years intensity of Somali CFE at the onset pentad

The aim of this picture is to show that in the early onset year, Somali CEF is very weak, which determine the weak precipitation over the eastern BOB
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

path0 = '/home/sun/data/composite/every_year_composite_array/'
fname = 'ERA5_composite_vwind_every_year_1980_2019.nc' # (40, 40days)

# Set area border for the Somali CEF
lat1  = 5
lat2  = -5
lon1  = 50
lon2  = 60

# Set time border to imply the onset period
time1 = 28
time2 = 32

# Read data
f0        = xr.open_dataset(path0 + fname).sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).isel(day=slice(time1, time2))
date_file = xr.open_dataset('/home/sun/data/onset_day_data/onsetdate.nc')
date      = date_file['bob_onset_date'].data

# === Calculate anomaly in each year ===
sml_anomoly = np.array([])
for yyyy in range(40):
    sml_anomoly = np.append(sml_anomoly, np.average(f0['v'].data[yyyy]))
#print(sml_anomoly)
#print(np.average(sml_anomoly))
sml_anomoly[:] -= np.average(sml_anomoly)
#print(sml_anomoly)

# === Calculate Linear Regression ===
x_train = np.array(date - np.average(date)).reshape((len(date - np.average(date)), 1))
y_train = np.array(sml_anomoly - np.average(sml_anomoly)).reshape((len(sml_anomoly - np.average(sml_anomoly)), 1))
lineModel = LinearRegression()
lineModel.fit(x_train, y_train)
Y_predict1 = lineModel.predict(x_train)

font = {'family': 'serif',
        'color':  'darkred',
        'weight': 'normal',
        'size': 16,
        }

plt.xlim((-20, 20))
plt.ylim((-4.5, 4.5))
plt.ylabel('Wind Speed (m/s)', fontdict=font)
plt.xlabel('Day', fontdict=font)
plt.scatter(date - np.average(date), sml_anomoly, marker='^', color='red')
plt.plot(x_train, Y_predict1, color='darkred', linewidth=1.5)
plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/onset_period_era5_somali_cef_intensity_with_dates.pdf')