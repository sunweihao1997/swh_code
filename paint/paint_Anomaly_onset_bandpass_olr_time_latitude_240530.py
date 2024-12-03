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

olr_f     = xr.open_dataset("/home/sun/data/composite/early_late_composite/ERA5_OLR_bandpass_early_late_composite.nc").sel(lon=lon_slice)

olr_early = np.average(olr_f['olr_early'].data, axis=2)
olr_late  = np.average(olr_f['olr_late'].data, axis=2)

# =========== plot the figure ===========
from matplotlib import cm
from matplotlib.colors import ListedColormap

# -------     figure    -----------
proj  =  ccrs.PlateCarree()
fig1    =  plt.figure(figsize=(20,10))
spec1   =  fig1.add_gridspec(nrows=1,ncols=2)

j  =  0

# ------       paint    ------------
row=0
olr_combined = [olr_early[:, ::-1], olr_late[:, ::-1]]
for col in range(2):
    ax = fig1.add_subplot(spec1[row,col])

    im  =  ax.contourf(olr_f.time.data, olr_f.lat.data[::-1], -1 * olr_combined[col].T, levels=np.linspace(-20, 20, 11), cmap='coolwarm', alpha=1, extend='both')

    ax.set_ylim((-10, 40))



    
# 加colorbar
fig1.subplots_adjust(top=0.8) 
cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
cb  =  fig1.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
cb.ax.tick_params(labelsize=20)

plt.savefig("/home/sun/paint/monsoon_onset_composite_ERA5/OLR_bandpass_lon_time.pdf")