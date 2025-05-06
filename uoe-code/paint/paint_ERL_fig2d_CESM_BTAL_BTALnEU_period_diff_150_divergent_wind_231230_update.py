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

# ====================== Calculate the divergence and divergent wind =======================

periodA = slice(1900, 1920) ; periodB = slice(1940, 1960)

data_p1 = file0.sel(time=periodA)
data_p2 = file0.sel(time=periodB)

btal_du  = np.average(data_p2['btal_ud'], axis=0)  - np.average(data_p1['btal_ud'], axis=0)
btal_dv  = np.average(data_p2['btal_vd'], axis=0)  - np.average(data_p1['btal_vd'], axis=0)
btal_div = np.average(data_p2['btal_div'], axis=0) - np.average(data_p1['btal_div'], axis=0)

btalneu_du  = np.average(data_p2['btalneu_ud'], axis=0)  - np.average(data_p1['btalneu_ud'], axis=0)
btalneu_dv  = np.average(data_p2['btalneu_vd'], axis=0)  - np.average(data_p1['btalneu_vd'], axis=0)
btalneu_div = np.average(data_p2['btalneu_div'], axis=0) - np.average(data_p1['btalneu_div'], axis=0)


# ===================== Function for painting =========================
def paint_jjas_diff(u, v, div, p, out_name, left_title, right_title, tlat):
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
    im1  =  ax.contourf(lon, tlat, div * 1e6, levels=np.linspace(-0.5, 0.5, 11), cmap=cmap, alpha=1, extend='both')

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #print(p.shape)
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.1)

    # Vector Map
    q  =  ax.quiver(lon, tlat, u, v, 
        regrid_shape=12, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=0.03,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.35,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    add_vector_legend(ax=ax, q=q, speed=0.1, fontsize=12, location=(0.775, 0), length=0.225, quiver_x=0.875)

    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left', fontsize=17.5)
    ax.set_title(right_title, loc='right', fontsize=17.5)
#
#    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')
#
    out_path = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/'
    plt.savefig(out_path + out_name)
    #plt.savefig('test.png', dpi=600)
#
def main():


    paint_jjas_diff(btal_du - btalneu_du, btal_dv - btalneu_dv, btal_div - btalneu_div    , file1['p_div'].data, out_name='ERL_fig2d_CESM_BTAL_BTALnEU_150_divergent_uv_div_period_difference.pdf',  left_title="(d)", right_title="Divergent Wind", tlat=lat)
    #paint_jjas_diff(btal_du              , btal_dv              , btal_div                , None, out_name='ERL_fig2d_CESM_BTAL_150_divergent_uv_div_period_difference.pdf',  left_title="BTAL", right_title="150 hPa", tlat=lat)
    #paint_jjas_diff(btalneu_du           , btalneu_dv           , btalneu_div             , None, out_name='ERL_fig2d_CESM_BTALnEU_150_divergent_uv_div_period_difference.pdf',  left_title="BTALnEU", right_title="150 hPa", tlat=lat)


if __name__ == '__main__':
    main()