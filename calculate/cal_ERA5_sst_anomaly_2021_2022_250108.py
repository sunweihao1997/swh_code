'''
2025-1-6
This script is to calculate the precipitation anomaly within 2021 and 2022

modify from /home/sun/swh_code/calculate/cal_ERA5_precipitation_anomaly_2021_2022_250106.py
'''
import xarray as xr
import numpy as np
import os

src_path     = "/home/sun/mydown/ERA5/monthly_single/"
src_filename = "yyyy_single_month.nc"

# 1. ================== Calculate the multi-year mean =====================
# 1.1 claim the averaged array
avg_pr = np.zeros((12, 721, 1440))

for yyyy in range(1980, 2024):
    filename0 = src_filename.replace("yyyy", str(int(yyyy)))

    f0        =  xr.open_dataset(src_path + filename0)
    #print(filename0)
    avg_pr    += (f0['skt'].data / (2024-1980)) 

print("Climatological value has been calculated!!")

# 2. =================== Calculate the anomaly for 2021 and 2022 ==============
f2021 = xr.open_dataset(src_path + src_filename.replace("yyyy", str(int(2021))))
f2022 = xr.open_dataset(src_path + src_filename.replace("yyyy", str(int(2022))))

anomaly2021 = f2021['skt'].data - avg_pr
anomaly2022 = f2022['skt'].data - avg_pr

# 2.1 mask the value
fmask  = xr.open_dataset("/home/sun/data/mask/ERA5_land_sea_mask.nc")
#print(anomaly2021.shape)
#print(fmask['lsm'].data.shape)
for t in range(12):
    anomaly2021[t, fmask['lsm'].data[0]>0.1] = np.nan
    anomaly2022[t, fmask['lsm'].data[0]>0.1] = np.nan

#print(np.nanmean(anomaly2021[5, :, 100]))

# 3. =================== Paint the anomaly ==================
from matplotlib import cm
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import sys
sys.path.append("/home/sun/swh_code/module/")
from module_sun import set_cartopy_tick

level0 = np.linspace(-1, 1, 11)
# -------   cartopy extent  -----
lonmin,lonmax,latmin,latmax  =  40,150,-15,35
extent     =  [lonmin,lonmax,latmin,latmax]

# -------     figure    -----------
proj  =  ccrs.PlateCarree()
fig1    =  plt.figure(figsize=(50,15))
spec1   =  fig1.add_gridspec(nrows=2,ncols=1)

# ------      paint    -----------
# First panel
row = 0 ; col = 0
ax = fig1.add_subplot(spec1[row,col],projection=proj)

# 设置刻度
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(40,150,7,dtype=int),yticks=np.linspace(-30,30,7,dtype=int),nx=1,ny=1,labelsize=20)

im  =  ax.contourf(f0.longitude.data, f0.latitude.data, anomaly2022[4] - anomaly2021[4], level0, cmap='coolwarm', alpha=1, extend='both')


# 海岸线
ax.coastlines(resolution='50m',lw=1.65)


ax.set_title("May", loc='left', fontsize=20)
ax.set_title("2022 - 2021", loc='right', fontsize=20)

# Second panel
row = 1 ; col = 0
ax = fig1.add_subplot(spec1[row,col],projection=proj)

# 设置刻度
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(40,150,7,dtype=int),yticks=np.linspace(-30,30,7,dtype=int),nx=1,ny=1,labelsize=20)

im  =  ax.contourf(f0.longitude.data, f0.latitude.data, anomaly2022[5] -anomaly2021[5], level0, cmap='coolwarm', alpha=1, extend='both')


# 海岸线
ax.coastlines(resolution='50m',lw=1.65)


ax.set_title("June", loc='left', fontsize=20)
ax.set_title("2022 - 2021", loc='right', fontsize=20)


# 加colorbar
fig1.subplots_adjust(top=0.8) 
cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
cb  =  fig1.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
cb.ax.tick_params(labelsize=10)


plt.savefig("/home/sun/paint/anomaly_2021_2022/skt_anomaly_2022_2021_diff.pdf", dpi=500)