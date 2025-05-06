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
from windspharm.xarray import VectorWind

# ======================= File Information ============================
# Here I select file saving single layer variable at 150 hPa

path_src = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
#file_src = 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_150_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc'
file_src = 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc'

# Coordination Information

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ======================================================================

# ====================== Calculate the divergence and divergent wind =======================

periodA = slice(1900, 1920) ; periodB = slice(1940, 1960)

data_p1 = file0.sel(time=periodA) ; data_p2 = file0.sel(time=periodB)

btal_u_p1    = data_p1['btal_u'].interp(lat=np.linspace(-90, 90, 121,))    ; btal_u_p2    = data_p2['btal_u'].interp(lat=np.linspace(-90, 90, 121,))
btal_v_p1    = data_p1['btal_v'].interp(lat=np.linspace(-90, 90, 121,))    ; btal_v_p2    = data_p2['btal_v'].interp(lat=np.linspace(-90, 90, 121,))

btalneu_u_p1 = data_p1['btalneu_u'].interp(lat=np.linspace(-90, 90, 121,)) ; btalneu_u_p2 = data_p2['btalneu_u'].interp(lat=np.linspace(-90, 90, 121,))
btalneu_v_p1 = data_p1['btalneu_v'].interp(lat=np.linspace(-90, 90, 121,)) ; btalneu_v_p2 = data_p2['btalneu_v'].interp(lat=np.linspace(-90, 90, 121,))

# Interpolate to equally spaced coordination


vector_btal_p1    = VectorWind(btal_u_p1, btal_v_p1)
vector_btal_p2    = VectorWind(btal_u_p2, btal_v_p2)

vector_btalneu_p1 = VectorWind(btalneu_u_p1, btalneu_v_p1)
vector_btalneu_p2 = VectorWind(btalneu_u_p2, btalneu_v_p2)

btal_du_p1, btal_dv_p1        =  vector_btal_p1.irrotationalcomponent()
btal_du_p2, btal_dv_p2        =  vector_btal_p2.irrotationalcomponent()

btalneu_du_p1, btalneu_dv_p1  =  vector_btalneu_p1.irrotationalcomponent()
btalneu_du_p2, btalneu_dv_p2  =  vector_btalneu_p2.irrotationalcomponent()

btal_div_p1                   =  vector_btal_p1.divergence()
btal_div_p2                   =  vector_btal_p2.divergence()

btalneu_div_p1                =  vector_btalneu_p1.divergence()
btalneu_div_p2                =  vector_btalneu_p2.divergence()

# The following is the period difference and the difference between the BTAL and BTALnEU

btal_du     =  np.average(btal_du_p2, axis=0)    - np.average(btal_du_p1, axis=0)
btalneu_du  =  np.average(btalneu_du_p2, axis=0) - np.average(btalneu_du_p1, axis=0)

btal_dv     =  np.average(btal_dv_p2, axis=0)    - np.average(btal_dv_p1, axis=0)
btalneu_dv  =  np.average(btalneu_dv_p2, axis=0) - np.average(btalneu_dv_p1, axis=0)

btal_div    =  np.average(btal_div_p2, axis=0)   - np.average(btal_div_p1, axis=0)
btalneu_div =  np.average(btalneu_div_p2, axis=0)- np.average(btalneu_div_p1, axis=0)

#print(np.min(btal_du_p2))
#print(np.max(btal_du_p2))

# ===================== Function for painting =========================
def paint_jjas_diff(u, v, div, p, out_name, left_title, right_title, tlat):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 10), subplot_kw={'projection': proj})


    # Tick setting
    # --- Set range ---
    #lonmin,lonmax,latmin,latmax  =  55, 105, 0, 40
    lonmin,lonmax,latmin,latmax  =  35, 155, 0, 70
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60, 100, 5,dtype=int), yticks=np.linspace(0, 40, 5, dtype=int),nx=1,ny=1,labelsize=15)


    # contourf for the geopotential height
    im1  =  ax.contourf(lon, tlat, div * 1e6, levels=np.linspace(-5, 5, 11), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, new_lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.1)

    # Vector Map
    q  =  ax.quiver(lon, tlat, u, v, 
        regrid_shape=12, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=0.9,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.35,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    add_vector_legend(ax=ax, q=q, speed=0.5, fontsize=12, location=(0.775, 0), length=0.225, quiver_x=0.875)

    #ax.set_ylabel("Influence of EU emission", fontsize=11)

#    ax.set_title(left_title, loc='left', fontsize=17.5)
#    ax.set_title(right_title, loc='right', fontsize=17.5)
#
#    # Add colorbar
#    plt.colorbar(im1, orientation='horizontal')
#
    out_path = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/'
    plt.savefig(out_path + out_name)
    #plt.savefig('test.png', dpi=600)
#
def main():


#    paint_jjas_diff(np.average(btal_u_p1['btal_u'], axis=0), np.average(btal_u_p1['btal_u'], axis=0), btal_div , file0['p_div'].data, out_name='ERL_fig2d_CESM_BTAL_BTALnEU_150_divuv_div_period_difference.pdf',  left_title="BTAL", right_title="150 hPa")
#    paint_jjas_diff(u_diff_BTAL              , v_diff_BTAL              , div_diff_BTAL                , ncfile['p_div'].data, out_name='ERL_fig2c_CESM_BTAL_150_uv_div_period_difference.pdf',  left_title="BTAL", right_title="150 hPa")
#    paint_jjas_diff(u_diff_noEU              , v_diff_noEU              , div_diff_noEU                , ncfile['p_div'].data,    out_name='ERL_fig2c_CESM_BTALnEU_150_uv_div_period_difference.pdf',  left_title="BTALnEU", right_title="150 hPa")

    # Test
    new_lat =  np.linspace(-90, 90, 121,)
    paint_jjas_diff(np.average(data_p1['btal_u'], axis=0), np.average(data_p1['btal_v'], axis=0), np.average(data_p1['btal_div'], axis=0) , None, out_name='ERL_fig2d_test.pdf',  left_title="BTAL", right_title="150 hPa", tlat=lat)
    paint_jjas_diff(np.average(data_p1['btal_u'].interp(lat=np.linspace(-90, 90, 121,)), axis=0), np.average(data_p1['btal_v'].interp(lat=np.linspace(-90, 90, 121,)), axis=0),  np.average(btal_div_p1, axis=0), None, out_name='ERL_fig2d_test2.pdf',  left_title="BTAL", right_title="150 hPa", tlat=new_lat)


if __name__ == '__main__':
    main()