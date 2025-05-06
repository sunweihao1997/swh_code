'''
2024-1-9
This script is to paint the difference between two periods in the streamfunction, meridional wind at 150 hPa
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
file_src = 'Aerosol_Research_CESM_BTAL_BTALnEU_150hPa_streamfunction_velocity_potential.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ================================ Painting ==============================================

def paint_jjas_diff(sf, v, p, pic_name, left_title):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(20, 15), subplot_kw={'projection': proj})

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  10,150,0,80
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(0,80,9,dtype=int),nx=1,ny=1,labelsize=10.5)

    # contourf for the meridional wind v
    im1  =  ax.contourf(lon, lat, v, levels=np.linspace(-1.2, 1.2, 25), cmap='coolwarm', alpha=1, extend='both')

    # contour for the streamfunction
    im2  =  ax.contour(lon, lat, sf, levels=np.linspace(-12, 12, 13), alpha=1, colors='k',)
    ax.clabel(im2, fontsize=6, inline=True)

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.1)


    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left', fontsize=12.5)
    ax.set_title('150 hPa', loc='right', fontsize=12.5)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/{}'.format(pic_name))
    #plt.savefig('test.png', dpi=600)

def paint_jjas_diff2(sf, w, p, pic_name, left_title):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(20, 15), subplot_kw={'projection': proj})

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  10,150,0,80
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(0,80,9,dtype=int),nx=1,ny=1,labelsize=10.5)

    # contourf for the meridional wind v
    im1  =  ax.contourf(lon, lat, w*100, levels=np.linspace(-0.8, 0.8, 17), cmap='coolwarm', alpha=1, extend='both')

    # contour for the streamfunction
    im2  =  ax.contour(lon, lat, sf, levels=np.linspace(-12, 12, 13), alpha=1, colors='k',)
    ax.clabel(im2, fontsize=6, inline=True)

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.1)


    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left', fontsize=12.5)
    ax.set_title('150 hPa', loc='right', fontsize=12.5)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/{}'.format(pic_name))
    #plt.savefig('test.png', dpi=600)

def main():
    # 1. Firstly, calculate difference between two periods for each experiment for the streamfunction
    ncfile  =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/Aerosol_Research_CESM_BTAL_BTALnEU_150hPa_streamfunction_velocity_potential.nc')

    ncfile0 = ncfile.sel(time=slice(1900, 1920))
    ncfile1 = ncfile.sel(time=slice(1940, 1960))


    sf_diff_BTAL = np.average(ncfile1['btal_sf'].data, axis=0) - np.average(ncfile0['btal_sf'].data, axis=0)
    sf_diff_noEU = np.average(ncfile1['btalneu_sf'].data, axis=0) - np.average(ncfile0['btalneu_sf'].data, axis=0)

    # This is the EU aerosol influence on the long-term trend
    sf_diff = sf_diff_BTAL - sf_diff_noEU

    
    # 2, Read and calculate meridional wind

    v_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_150_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc'

    v_file = xr.open_dataset(v_path)

    v_file0 = v_file.sel(time=slice(1900, 1920))
    v_file1 = v_file.sel(time=slice(1940, 1960))

    v_diff_btal    = np.average(v_file1['btal_v'].data, axis=0)    - np.average(v_file0['btal_v'].data, axis=0)
    v_diff_btalneu = np.average(v_file1['btalneu_v'].data, axis=0) - np.average(v_file0['btalneu_v'].data, axis=0)

    v_diff         = v_diff_btal - v_diff_btalneu

    w_diff_BTAL = np.average(v_file1['btal_w'].data, axis=0)    - np.average(v_file0['btal_w'].data, axis=0)
    w_diff_noEU = np.average(v_file1['btalneu_w'].data, axis=0) - np.average(v_file0['btalneu_w'].data, axis=0)
    
    w_diff         = w_diff_BTAL - w_diff_noEU
    print(v_file['btal_w'].data)
    print(np.min(w_diff_BTAL))
    print(np.max(w_diff))

    paint_jjas_diff(sf_diff/1e5, v_diff, None, "ERL_fig3_CESM_BTAL_streamfunction_meridional_wind_period_diff_150.pdf", '(a)')
    paint_jjas_diff2(sf_diff/1e5, w_diff, None, "ERL_fig3_type2_rp_v_to_w_CESM_BTAL_streamfunction_meridional_wind_period_diff_150.pdf", '(a)')


if __name__ == '__main__':
    main()