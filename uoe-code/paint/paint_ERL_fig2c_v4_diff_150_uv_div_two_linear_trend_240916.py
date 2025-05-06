'''
2023-12-3
This script is to paint the difference between two periods on the Z, V at 200 hPa, from BTAL model output

The type2 means: Replace the divergence with the 500 hPa Omega

v3:
change to linear trend

v4:
move to huaibei server
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
sys.path.append('/home/sun/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend
#import cmasher as cmr
import seaborn as sns
from scipy.ndimage import gaussian_filter

# ================================ File location =========================================

path_src = '/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_src = 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ================================ Painting ==============================================

def calculate_linear_trend(start, end, input_array, varname):
    from scipy.stats import linregress

    time_dim, lat_dim, lon_dim = input_array.sel(time=slice(start, end))[varname].shape

    trend_data = np.zeros((lat_dim, lon_dim))
    p_data     = np.zeros((lat_dim, lon_dim))

    input_data = input_array.sel(time=slice(start, end))[varname].data
    #print(input_data.shape)

    for i in range(lat_dim):
        for j in range(lon_dim):
            #print(linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j]))
            slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j])
            trend_data[i, j] = slope
            p_data[i, j]    = p_value

    return trend_data, p_value

def paint_jjas_diff(u, v, div, p, out_name, left_title, right_title):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 10), subplot_kw={'projection': proj})

    cmap = sns.color_palette("Spectral", as_cmap=True)

    # Tick setting
    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  55, 125, 5, 40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60, 130, 8,dtype=int), yticks=np.linspace(0, 40, 5, dtype=int),nx=1,ny=1,labelsize=25)

    # contourf for the geopotential height
    im1  =  ax.contourf(lon, lat, div, levels=np.linspace(-1, 1, 11), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='110m', lw=1.5)

    # Vector Map
    q  =  ax.quiver(lon, lat, u, v, 
        regrid_shape=12, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=0.05,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.35,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    #add_vector_legend(ax=ax, q=q, speed=0.1, fontsize=12, location=(0.775, 0), length=0.225, quiver_x=0.875)

    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left', fontsize=25)
    ax.set_title(right_title, loc='right', fontsize=25)

    # Add colorbar
    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im1, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    cb.ax.set_xticks(np.linspace(-1, 1, 11))
    cb.ax.tick_params(labelsize=15)

    out_path = '/home/sun/paint/ERL/'
    plt.savefig(out_path + out_name)
    #plt.savefig('test.png', dpi=600)

def main():
    # 1. Firstly, calculate difference between two periods for each experiment    
    # ================================== 150 hPa ===========================================
    ncfile  =  xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_150_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc').sel(time=slice(1901, 1955))

    p1 = 1901 ; p2 = 1955
    u_con, u_p_con = calculate_linear_trend(p1, p2, ncfile, 'btal_u')
    v_con, v_p_con = calculate_linear_trend(p1, p2, ncfile, 'btal_v')
    u_neu, u_p_neu = calculate_linear_trend(p1, p2, ncfile, 'btalneu_u')
    v_neu, v_p_neu = calculate_linear_trend(p1, p2, ncfile, 'btalneu_v')
    div_con, div_p_con = calculate_linear_trend(p1, p2, ncfile,    'btal_div')
    div_neu, div_p_neu = calculate_linear_trend(p1, p2, ncfile,    'btalneu_div')
    

    paint_jjas_diff(10*gaussian_filter((u_con - u_neu), sigma=0.5), 10*gaussian_filter((v_con - v_neu), sigma=0.5), 1e8*gaussian_filter((div_con - div_neu), sigma=1), div_p_con, out_name='ERL_fig2c_v3_CESM_BTAL_BTALnEU_150_uv_div_linear_trend.pdf',  left_title="(c)", right_title="150 hPa")
    #paint_jjas_diff(u_diff_BTAL              , v_diff_BTAL              , div_diff_BTAL                , ncfile['p_div'].data, out_name='ERL_fig2c_CESM_BTAL_150_uv_div_period_difference.pdf',  left_title="BTAL", right_title="150 hPa")
    #paint_jjas_diff(u_diff_noEU              , v_diff_noEU              , div_diff_noEU                , ncfile['p_div'].data,    out_name='ERL_fig2c_CESM_BTALnEU_150_uv_div_period_difference.pdf',  left_title="BTALnEU", right_title="150 hPa")

if __name__ == '__main__':
    main()