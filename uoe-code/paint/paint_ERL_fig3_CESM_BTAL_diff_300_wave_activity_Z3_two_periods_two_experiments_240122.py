'''
2023-12-7 modified from paint_CESM_BTAL_diff_200_wave_activity_Z3_two_periods_two_experiments_231205_new.py
This script is to paint the difference between two periods on the Z, wave activity at 300 hPa
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
from module_sun import check_path, add_vector_legend, cal_xydistance

import seaborn as sns

# ================================ File location =========================================

path_src = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_src = 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# ================================ Painting ==============================================

def paint_jjas_diff(u, v, z, p, pic_name):
    '''This function paint the Diff aerosol JJA'''
    proj       =  ccrs.PlateCarree()
    ax         =  plt.subplot(projection=proj)

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    cmap = sns.diverging_palette(145, 300, s=60, as_cmap=True)

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  10,150,10,70
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(10,70,4,dtype=int),nx=1,ny=1,labelsize=12)

    # contourf for the geopotential height
    level0 = np.array([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5,])
    im1  =  ax.contourf(lon, lat, z, levels=level0, cmap='coolwarm', alpha=1, extend='both')

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.5)

    # Vector Map
    q  =  ax.quiver(lon, lat, u, v, 
        regrid_shape=10, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=0.002,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.7,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    #add_vector_legend(ax=ax, q=q, speed=0.25)

    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title('(b)', loc='left',      fontsize=12)
    ax.set_title('300 hPa', loc='right', fontsize=12)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im1, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    cb.ax.set_xticks(level0)
    cb.ax.tick_params(labelsize=7.5)


    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/{}'.format(pic_name))
    #plt.savefig('test.png', dpi=600)

def main():
    # 1. Firstly, calculate difference between two periods for each experiment
    ncfile  =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc')

    ncfile0 = ncfile.sel(time=slice(1900, 1920))
    ncfile1 = ncfile.sel(time=slice(1940, 1960))


    z_diff_BTAL = np.average(ncfile1['btal_z'].data, axis=0) - np.average(ncfile0['btal_z'].data, axis=0)
    z_diff_noEU = np.average(ncfile1['btalneu_z'].data, axis=0) - np.average(ncfile0['btalneu_z'].data, axis=0)

    # This is the EU aerosol influence on the long-term trend
    z_diff = z_diff_BTAL - z_diff_noEU
    
    # 2, Read and calculate wave activity

    wave_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/wave_activity/'

    Fx = xr.open_dataset(wave_path + "BTAL_BTALnEU_diff_Z3_for_TN2001-Fx.monthly.period2.nc").sel(level=300)
    Fy = xr.open_dataset(wave_path + "BTAL_BTALnEU_diff_Z3_for_TN2001-Fy.monthly.period2.nc").sel(level=300)

    #print(Fx)
    #u_diff = (BTAL_Fx2['Fx'] - BTAL_Fx1['Fx']) - (BTALnEU_Fx2['Fx'] - BTALnEU_Fx1['Fx'])
    #v_diff = (BTAL_Fy2['Fy'] - BTAL_Fy1['Fy']) - (BTALnEU_Fy2['Fy'] - BTALnEU_Fy1['Fy'])

    u_diff = Fx['Fx']
    v_diff = Fy['Fy']

    # 3. Calculate the divergence of the wave activity 
    lat = Fx['lat'].data
    lon = Fy['lon'].data

    disy,disx,location = cal_xydistance(lat,lon)

    v_diff_dy  =  np.gradient(v_diff, location, axis=0)
    u_diff_dx  =  np.zeros(v_diff_dy.shape)

    for yy in range(0,v_diff_dy.shape[0]):
        u_diff_dx[yy, :]  =  np.gradient(u_diff[yy, :], disx[yy], axis=0)


    paint_jjas_diff(u_diff, v_diff, (u_diff_dx + v_diff_dy) * 1e7, ncfile['p_z'].data, pic_name="ERL_fig3_wave_activity_period_diff_300_Z3_new_calculation_method.pdf")

#    u_diff = (BTALnEU_Fx2['Fx'] - BTALnEU_Fx1['Fx'])
#    v_diff = (BTALnEU_Fy2['Fy'] - BTALnEU_Fy1['Fy'])
#    paint_jjas_diff(u_diff, v_diff, z_diff_noEU, ncfile['p_z'].data, pic_name="EUI_CESM_BTALnEU_wave_activity_period_diff_200_Z3.pdf")
#
#    u_diff = (BTAL_Fx2['Fx'] - BTAL_Fx1['Fx']) - (BTALnEU_Fx2['Fx'] - BTALnEU_Fx1['Fx'])
#    v_diff = (BTAL_Fy2['Fy'] - BTAL_Fy1['Fy']) - (BTALnEU_Fy2['Fy'] - BTALnEU_Fy1['Fy'])
#    paint_jjas_diff(u_diff, v_diff, z_diff, ncfile['p_z'].data, pic_name="EUI_CESM_BTAL_BTALnEU_wave_activity_period_diff_200_Z3.pdf")

if __name__ == '__main__':
    main()