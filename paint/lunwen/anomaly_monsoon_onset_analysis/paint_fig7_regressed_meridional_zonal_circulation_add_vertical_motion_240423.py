'''
2024-4-23
This script is to plot the regressed FMA 3-d circulation over tropical Indian OCean and BOB
'''
from windspharm.xarray import VectorWind
import numpy as np
import xarray as xr
from matplotlib import projections
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cartopy
import cmasher as cmr
import os
import concurrent.futures

data_path = '/home/sun/data/ERA5_data_monsoon_onset/regression/'
data_name = 'ERA5_regression_OLR-remove-SLP_zonal_tropical_circulation.nc'

f0        = xr.open_dataset(data_path + data_name)

lev       = f0.level.data
lon       = f0.longitude.data

def plot_image_zonal(u, w, level=0):
    # ========== Painting ===============
    fig1, ax    =  plt.subplots(figsize=(15,15),)    

    # Shading for precipitation
    level  =  np.linspace(-1, 1, 11)
    im  =  ax.contourf(lon, lev, w[::-1] * -1000, cmap='coolwarm', alpha=1, extend='both') 

    # Set the axis tick
    ax.set_xticks(np.linspace(40, 140, 6, dtype=int))
    ax.set_yticks(np.linspace(1000, 200, 5, dtype=int))

    ax.tick_params(axis='both',labelsize=25)

    ax.set_xlim((30, 150))
    ax.set_ylim((1000, 200))

    q  =  ax.quiver(lon, lev[::-1], u, w * -1000, 
            angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=1.5,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)    


    # 3. Color Bar
    fig1.subplots_adjust(top=0.8) 
    cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig1.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=25) 

    plt.savefig('/home/sun/paint/lunwen/anomoly_analysis/v1_fig7_FMA_zonal_diff.pdf')

if __name__ == '__main__':
    plot_image_zonal(f0['rc_u'].data, f0['rc_w'].data)