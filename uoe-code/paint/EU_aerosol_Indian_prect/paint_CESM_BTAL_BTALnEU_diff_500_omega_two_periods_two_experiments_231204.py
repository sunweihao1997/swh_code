'''
2023-12-3
This script is to paint the difference between two periods on the Z, V at 200 hPa, from BTAL model output
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
file_src = 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_and_ttest_1945_1960.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ================================ Painting ==============================================

def paint_jjas_diff(omega, p):
    '''This function paint the Diff aerosol JJA'''
    proj       =  ccrs.PlateCarree()
    ax         =  plt.subplot(projection=proj)

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  0,150,-10,80
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(-10,70,9,dtype=int),nx=1,ny=1,labelsize=10.5)

    # contourf for the geopotential height
    im1  =  ax.contourf(lon, lat, omega * 1000, levels=np.linspace(-5, 5, 11), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='110m', lw=1.1)

    ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title('1901-1920 to 1941-1960', fontsize=12.5)
    ax.set_title('500 hPa', loc='right', fontsize=12.5)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    #plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/EUI_EU_emission_influence_long-term_trend_500hPa_omega.pdf')
    plt.savefig('test.png', dpi=600)

def main():
    # 1. Firstly, calculate difference between two periods for each experiment
    ncfile  =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_and_ttest_1945_1960.nc')

    ncfile0 = ncfile.sel(time=slice(1900, 1920))
    ncfile1 = ncfile.sel(time=slice(1940, 1960))

    w_diff_BTAL = np.average(ncfile1['btal_w'].data, axis=0) - np.average(ncfile0['btal_w'].data, axis=0)
    w_diff_noEU = np.average(ncfile1['btalneu_w'].data, axis=0) - np.average(ncfile0['btalneu_w'].data, axis=0)


    # This is the EU aerosol influence on the long-term trend
    #w_diff = w_diff_BTAL - w_diff_noEU
    w_diff = w_diff_BTAL

    paint_jjas_diff(w_diff, ncfile['p_w'].data)

if __name__ == '__main__':
    main()