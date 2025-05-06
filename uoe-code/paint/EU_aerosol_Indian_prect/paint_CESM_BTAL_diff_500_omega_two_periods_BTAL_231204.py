'''
2023-12-4
This script is to paint the difference between two periods on the OMEGA at 500 hPa, from BTAL model output and difference among the experiments
'''
import xarray as xr
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# ================================ File location =========================================

path_src = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_src = 'EUI_CESM_diff_BTAL_1900_1960_200hPa_Z_UV_500hPa_Omega.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ================================ Painting ==============================================

def paint_jjas_diff_BTAL(omega, p):
    '''This function paint the Diff aerosol JJA'''
    proj       =  ccrs.PlateCarree()
    ax         =  plt.subplot(projection=proj)

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  0,150,-10,80
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=[0, 150, 0, 80],xticks=np.linspace(0,150,6,dtype=int),yticks=np.linspace(0,80,9,dtype=int),nx=1,ny=1,labelsize=10)

    # contourf for the geopotential height
    im1  =  ax.contourf(lon, lat, omega * 1000, levels=np.linspace(-5, 5, 11), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    sp  =  ax.contourf(lon, lat, p, levels=[0.1, 1], colors='none', hatches=['...'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='110m', lw=1.1)

    #ax.set_ylabel("BTAL long-term changes", fontsize=11)

    ax.set_title('1901-1920 to 1941-1960', fontsize=15)
    ax.set_title('OMEGA', loc='left', fontsize=15)
    ax.set_title('CESM', loc='right', fontsize=15)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/EUI_BTAL_OMEGA_diff_1900_1960.pdf')
    #plt.savefig('test.png', dpi=600)

def main():
    # 1. Firstly, calculate difference between two periods for each experiment
    ncfile  =  xr.open_dataset(path_src + file_src)

    ensemble_omega_avg = np.average(ncfile['diff_w'], axis=0)

    paint_jjas_diff_BTAL(omega=ensemble_omega_avg, p=ncfile['stipple_w'].data)

if __name__ == '__main__':
    main()