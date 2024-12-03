'''
2023-5-30
This script is to plot the time-latitude cross-section for the bandpass OLR
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

# Calculate the zonal mean
lon_slice = slice(80, 100)

olr_f     = xr.open_dataset("/home/sun/data/composite/early_late_composite/ERA5_precipitation_bandpass_3080_early_late_composite.nc").sel(lon=lon_slice)

olr_early = np.average(olr_f['tp_early'].data, axis=2)
olr_late  = np.average(olr_f['tp_late'].data, axis=2)

print(np.nanmax(olr_late))
# =========== plot the figure ===========
from matplotlib import cm
from matplotlib.colors import ListedColormap

# -------     figure    -----------
proj  =  ccrs.PlateCarree()
fig1    =  plt.figure(figsize=(30,12))
spec1   =  fig1.add_gridspec(nrows=1,ncols=2)

j  =  0

# ------       paint    ------------
row=0
olr_combined = [olr_early[10:, ::-1], olr_late[10:, ::-1]]
for col in range(2):
    ax = fig1.add_subplot(spec1[row,col])

    im  =  ax.contourf(olr_f.time_composite.data[10:], olr_f.lat.data[::-1], olr_combined[col].T, levels=np.linspace(0.5, 3, 11), cmap='Blues', alpha=1,)

    ax.set_ylim((-5, 20))

    ax.tick_params(axis='x', labelsize=32)
    ax.tick_params(axis='y', labelsize=27)

    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels(["EQ", "10N", "20N"],)

    ax.set_xticks([-20, -15, -10, -5, 0, 5])
    ax.set_xticklabels(["D-20", "D-15", "D-10", "D-5", "D0", "D+5"])
    
    ax.plot([0, 0], [-5, 20], linewidth=3, linestyle='--', color='grey', alpha=0.6)



    
# 加colorbar
#fig1.subplots_adjust(top=0.8) 
#cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
#cb  =  fig1.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
#cb.ax.tick_params(labelsize=25)

plt.savefig("/home/sun/paint/phd/phd_C5_fig2_v0_bandpass_pr_time_lat_nocolor_map.pdf")