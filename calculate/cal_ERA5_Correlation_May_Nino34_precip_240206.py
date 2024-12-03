'''
2024-2-6
This script is to calculate and plot the correlation between the May Nino34 and Precipitation
'''
import xarray as xr
import numpy as np
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import scipy.stats

module_path = '/home/sun/mycode/module'
sys.path.append(module_path)

from module_sun_new import set_cartopy_tick

# ==== Data Process ====

data_path = '/home/sun/data/'
f0        = xr.open_dataset(data_path + 'ERA5_single_sst_tp_May_1980_2014.nc')

#print(f0.longitude.data) # 0-365

# ---- 1. Nino34 index ----

f0_Nino34 = f0.sel(latitude=slice(5, -5), longitude=slice(190, 240))

Nino34    = np.array([])

for i in range(35):
    Nino34= np.append(Nino34, np.nanmean(f0_Nino34['sst'].data[i]))

Nino34    = Nino34 - np.average(Nino34) # It is the Nino34 for the May

# ---- 2. Correlation ----

corre     = np.zeros((721, 1440))
p_value   = np.zeros((721, 1440))

for i in range(721):
    for j in range(1440):
        pearson_r     = scipy.stats.pearsonr(Nino34, (f0['tp'].data[:, i, j] - np.average(f0['tp'].data[:, i, j])))    
        corre[i, j]   = pearson_r[0]
        p_value[i, j] = pearson_r[1]

# ==== Save to the array ====

ncfile  =  xr.Dataset(
    {
        "corre":   (["lat", "lon"], corre),
        "p_value": (["lat", "lon"], p_value),
    },
    coords={
        "lat":    (["lat"],    f0['latitude'].data),
        "lon":    (["lon"],    f0['longitude'].data),
    },
        )

# ---- 1.1.1 Add attributes ----

ncfile['corre'].attrs['description'] = 'Pearson correlation between ensemble-averaged Nino34 and precipitation in May.'

ncfile.attrs['description'] = 'Create on 2024-2-5 on the Huaibei Server. This netcdf file saves Pearson correlation between ensemble-averaged Nino34 and precipitation in May.'

ncfile.to_netcdf('/home/sun/data/ERA5_correlation_May_precipitation_Nino34.nc')