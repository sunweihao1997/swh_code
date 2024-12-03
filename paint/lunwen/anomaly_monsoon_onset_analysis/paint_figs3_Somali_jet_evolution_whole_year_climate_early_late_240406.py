'''
2024-4-6
This script is to plot the evolution of the intensity of the Somali CEF. The picture should comprise the climatological and early/late year mean

The intention is to see whether the evolution of the Somali-CEF give different characteristics under different condition
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

data_path  = '/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5/single/'
data_namec = '10m_v_component_of_wind_climatic_daily.nc'
data_namee = '10m_v_component_of_wind_climatic_daily_year_early.nc'
data_namel = '10m_v_component_of_wind_climatic_daily_year_late.nc'

lat_slice = slice(10, 0) ; lon_slice = slice(50, 60)

file_early = xr.open_dataset(data_path + data_namee).sel(lat=lat_slice, lon=lon_slice)
file_late  = xr.open_dataset(data_path + data_namel).sel(lat=lat_slice, lon=lon_slice)
file_clima = xr.open_dataset(data_path + data_namec).sel(lat=lat_slice, lon=lon_slice)

cef_clima  = np.zeros((73))
cef_early  = np.zeros((73))
cef_late   = np.zeros((73))

for tt in range(73):
    cef_clima[tt] = np.average(file_clima['v10'].data[tt*5:tt*5+5])
    cef_late[tt]  = np.average(file_late['v10'].data[tt*5:tt*5+5])
    cef_early[tt] = np.average(file_early['v10'].data[tt*5:tt*5+5])

# Start plot
fig, ax = plt.subplots()
ax.plot(np.linspace(1, 30, 30), cef_clima[:30], 'k')
ax.plot(np.linspace(1, 30, 30), cef_early[:30], 'b--')
ax.plot(np.linspace(1, 30, 30), cef_late[:30],  'r--')

plt.savefig('cef.png')