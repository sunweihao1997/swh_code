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
import cmasher as cmr

# ================================ File location =========================================

path_src = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_src = 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ================================ Painting ==============================================

def paint_jjas_diff(u, v, div, p, out_name, left_title, right_title):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 10), subplot_kw={'projection': proj})

    cmap = 'coolwarm'

    # Tick setting
    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  55, 125, 0, 40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60, 130, 8,dtype=int), yticks=np.linspace(0, 40, 5, dtype=int),nx=1,ny=1,labelsize=25)


    # contourf for the geopotential height
    im1  =  ax.contourf(lon, lat, div * 1e6, levels=np.linspace(-0.5, 0.5, 11), cmap=cmap, alpha=1, extend='both')

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='110m', lw=1.5)

    # Vector Map
    q  =  ax.quiver(lon, lat, u, v, 
        regrid_shape=12, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=0.15,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.35,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    #add_vector_legend(ax=ax, q=q, speed=0.5, fontsize=12, location=(0.775, 0), length=0.225, quiver_x=0.875)

    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left', fontsize=17.5)
    ax.set_title(right_title, loc='right', fontsize=17.5)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    out_path = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/'
    plt.savefig(out_path + out_name)
    #plt.savefig('test.png', dpi=600)

def main():
    # 1. Firstly, calculate difference between two periods for each experiment
    # ================================== 200 hPa ===========================================
    ncfile  =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc')

    ncfile0 = ncfile.sel(time=slice(1900, 1920))
    ncfile1 = ncfile.sel(time=slice(1940, 1960))


    u_diff_BTAL = np.average(ncfile1['btal_u'].data, axis=0)    - np.average(ncfile0['btal_u'].data, axis=0)
    u_diff_noEU = np.average(ncfile1['btalneu_u'].data, axis=0) - np.average(ncfile0['btalneu_u'].data, axis=0)

    v_diff_BTAL = np.average(ncfile1['btal_v'].data, axis=0)    - np.average(ncfile0['btal_v'].data, axis=0)
    v_diff_noEU = np.average(ncfile1['btalneu_v'].data, axis=0) - np.average(ncfile0['btalneu_v'].data, axis=0)

    div_diff_BTAL = np.average(ncfile1['btal_div'].data, axis=0)    - np.average(ncfile0['btal_div'].data, axis=0)
    div_diff_noEU = np.average(ncfile1['btalneu_div'].data, axis=0) - np.average(ncfile0['btalneu_div'].data, axis=0)

    

    paint_jjas_diff(u_diff_BTAL - u_diff_noEU, v_diff_BTAL - v_diff_noEU, div_diff_BTAL - div_diff_noEU, ncfile['p_div'].data, out_name='ERL_fig2c_CESM_BTAL_BTALnEU_200_uv_div_period_difference.pdf', left_title="BTAL - BTALnEU", right_title="200 hPa")
    paint_jjas_diff(u_diff_BTAL              , v_diff_BTAL              , div_diff_BTAL                , ncfile['p_div'].data, out_name='ERL_fig2c_CESM_BTAL_200_uv_div_period_difference.pdf',         left_title="BTAL", right_title="200 hPa")
    paint_jjas_diff(u_diff_noEU              , v_diff_noEU              , div_diff_noEU             , ncfile['p_div'].data,    out_name='ERL_fig2c_CESM_BTALnEU_200_uv_div_period_difference.pdf',      left_title="BTALnEU", right_title="200 hPa")
    
    # ================================== 150 hPa ===========================================
    ncfile  =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_150_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc')

    ncfile0 = ncfile.sel(time=slice(1900, 1920))
    ncfile1 = ncfile.sel(time=slice(1940, 1960))


    u_diff_BTAL = np.average(ncfile1['btal_u'].data, axis=0)    - np.average(ncfile0['btal_u'].data, axis=0)
    u_diff_noEU = np.average(ncfile1['btalneu_u'].data, axis=0) - np.average(ncfile0['btalneu_u'].data, axis=0)

    v_diff_BTAL = np.average(ncfile1['btal_v'].data, axis=0)    - np.average(ncfile0['btal_v'].data, axis=0)
    v_diff_noEU = np.average(ncfile1['btalneu_v'].data, axis=0) - np.average(ncfile0['btalneu_v'].data, axis=0)

    div_diff_BTAL = np.average(ncfile1['btal_div'].data, axis=0)    - np.average(ncfile0['btal_div'].data, axis=0)
    div_diff_noEU = np.average(ncfile1['btalneu_div'].data, axis=0) - np.average(ncfile0['btalneu_div'].data, axis=0)

    

    paint_jjas_diff(u_diff_BTAL - u_diff_noEU, v_diff_BTAL - v_diff_noEU, div_diff_BTAL - div_diff_noEU, ncfile['p_div'].data, out_name='ERL_fig2c_CESM_BTAL_BTALnEU_150_uv_div_period_difference.pdf',  left_title="(c)", right_title="150 hPa")
    paint_jjas_diff(u_diff_BTAL              , v_diff_BTAL              , div_diff_BTAL                , ncfile['p_div'].data, out_name='ERL_fig2c_CESM_BTAL_150_uv_div_period_difference.pdf',  left_title="BTAL", right_title="150 hPa")
    paint_jjas_diff(u_diff_noEU              , v_diff_noEU              , div_diff_noEU                , ncfile['p_div'].data, out_name='ERL_fig2c_CESM_BTALnEU_150_uv_div_period_difference.pdf',  left_title="BTALnEU", right_title="150 hPa")

if __name__ == '__main__':
    main()