'''
2023-12-30
This script serves for the Fig2d in ERL, containing divergent wind and divergence(shading)
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
import seaborn as sns
from scipy.ndimage import gaussian_filter


# ======================= File Information ============================
# Here I select file saving single layer variable at 150 hPa

path_src = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_src2= 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_150_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc'
file_src = 'Aerosol_Research_CESM_divergent_wind_divergence_BTAL_BTALnEU_JJAS.nc'

# Coordination Information

file0  =  xr.open_dataset(path_src + file_src)
file1  =  xr.open_dataset(path_src + file_src2)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ======================================================================

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

# ====================== Calculate the divergence and divergent wind =======================

#periodA = slice(1900, 1920) ; periodB = slice(1940, 1960)
#
#data_p1 = file0.sel(time=periodA)
#data_p2 = file0.sel(time=periodB)
#
#btal_du  = np.average(data_p2['btal_ud'], axis=0)  - np.average(data_p1['btal_ud'], axis=0)
#btal_dv  = np.average(data_p2['btal_vd'], axis=0)  - np.average(data_p1['btal_vd'], axis=0)
#btal_div = np.average(data_p2['btal_div'], axis=0) - np.average(data_p1['btal_div'], axis=0)
#
#btalneu_du  = np.average(data_p2['btalneu_ud'], axis=0)  - np.average(data_p1['btalneu_ud'], axis=0)
#btalneu_dv  = np.average(data_p2['btalneu_vd'], axis=0)  - np.average(data_p1['btalneu_vd'], axis=0)
#btalneu_div = np.average(data_p2['btalneu_div'], axis=0) - np.average(data_p1['btalneu_div'], axis=0)

p1 = 1901 ; p2 = 1955
u_con, u_p_con = calculate_linear_trend(p1, p2, file0, 'btal_ud')
v_con, v_p_con = calculate_linear_trend(p1, p2, file0, 'btal_vd')
u_neu, u_p_neu = calculate_linear_trend(p1, p2, file0, 'btalneu_ud')
v_neu, v_p_neu = calculate_linear_trend(p1, p2, file0, 'btalneu_vd')
w_con, w_p_con = calculate_linear_trend(p1, p2, file1, 'btal_div')
w_neu, w_p_neu = calculate_linear_trend(p1, p2, file1, 'btalneu_div')


# ===================== Function for painting =========================
def paint_jjas_diff(u, v, w, p, out_name, left_title, right_title, tlat):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 10), subplot_kw={'projection': proj})

    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    # Tick setting
    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  55, 125, 5, 40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60, 130, 8,dtype=int), yticks=np.linspace(0, 40, 5, dtype=int),nx=1,ny=1,labelsize=25)


    # contourf for the geopotential height
    im1  =  ax.contourf(lon, tlat, w, levels=np.linspace(-1, 1, 11), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #print(p.shape)
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['/'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.5)

    # Vector Map
    q  =  ax.quiver(lon, tlat, u, v, 
        regrid_shape=12, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=0.075,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.35,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    add_vector_legend(ax=ax, q=q, speed=0.1, fontsize=12, location=(0.775, 0), length=0.225, quiver_x=0.875)

    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left',   fontsize=25)
    ax.set_title(right_title, loc='right', fontsize=25)
#
#    # ========= add colorbar =================
#    fig.subplots_adjust(top=0.8) 
#    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
#    cb  =  fig.colorbar(im1, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
#    cb.ax.set_xticks(np.linspace(-1, 1, 11))
#    cb.ax.tick_params(labelsize=15)
    plt.colorbar(im1, orientation='horizontal')
#
    out_path = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/'
    plt.savefig(out_path + out_name)
    #plt.savefig('test.png', dpi=600)
#
def main():


    paint_jjas_diff(1e2*gaussian_filter((u_con - u_neu), sigma=0.5), 1e2*gaussian_filter((v_con - v_neu), sigma=0.5), 1e8*gaussian_filter((w_con - w_neu), sigma=0.8), w_p_con, out_name='ERL_fig2d_v3_CESM_BTAL_BTALnEU_150_divergent_uv_div_linear_trend.pdf',  left_title="(d)", right_title="Divergent Wind", tlat=lat)
    #paint_jjas_diff(btal_du              , btal_dv              , btal_div                , None, out_name='ERL_fig2d_CESM_BTAL_150_divergent_uv_div_period_difference.pdf',  left_title="BTAL", right_title="150 hPa", tlat=lat)
    #paint_jjas_diff(btalneu_du           , btalneu_dv           , btalneu_div             , None, out_name='ERL_fig2d_CESM_BTALnEU_150_divergent_uv_div_period_difference.pdf',  left_title="BTALnEU", right_title="150 hPa", tlat=lat)


if __name__ == '__main__':
    main()