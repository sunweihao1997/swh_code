'''
2024-7-22
This script is to paint the difference between two periods in the streamfunction, meridional wind at 200 hPa

Linear trend
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
file_src = 'Aerosol_Research_CESM_BTAL_BTALnEU_200hPa_streamfunction_velocity_potential.nc'

# ========================================================================================

file0  =  xr.open_dataset(path_src + file_src)
lat    =  file0.lat.data
lon    =  file0.lon.data

# =========== Calculate linear trend ==============
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

    return trend_data, p_data

def calculate_linear_trend_diff(start, end, input_array, varname1, varname2):
    '''This calculate the linear trend difference between 2 given data'''
    from scipy.stats import linregress

    time_dim, lat_dim, lon_dim = input_array.sel(time=slice(start, end))[varname1].shape

    trend_data = np.zeros((lat_dim, lon_dim))
    p_data     = np.zeros((lat_dim, lon_dim))

    input_data1 = input_array.sel(time=slice(start, end))[varname1].data
    input_data2 = input_array.sel(time=slice(start, end))[varname2].data
    #print(input_data.shape)

    for i in range(lat_dim):
        for j in range(lon_dim):
            #print(linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j]))
            slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1, time_dim, time_dim), input_data1[:, i, j] - input_data2[:, i, j])
            trend_data[i, j] = slope
            p_data[i, j]    = p_value

    return trend_data, p_data

# ================================ Painting ==============================================

def paint_jjas_diff(sf, v, p, pic_name, left_title):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(20, 10), subplot_kw={'projection': proj})

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  0,150,10,65
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(0,80,9,dtype=int),nx=1,ny=1,labelsize=12)

    # contourf for the meridional wind v
    im1  =  ax.contourf(lon, lat, v, levels=np.linspace(-2.5, 2.5, 11), cmap='coolwarm', alpha=0.8, extend='both')

    # contour for the streamfunction
    im2  =  ax.contour(lon, lat, sf, levels=np.linspace(-2.5, 2.5, 11), alpha=1, colors='k',)
    ax.clabel(im2, fontsize=5, inline=True)

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    #print(p.shape)
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['//'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.1)


    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left',         fontsize=12)
    ax.set_title('Stream-function', loc='right', fontsize=12)

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
    lonmin,lonmax,latmin,latmax  =  10,150,10,70
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(10,70,4,dtype=int),nx=1,ny=1,labelsize=12)

    # contourf for the meridional wind v
    im1  =  ax.contourf(lon, lat, w*100, levels=np.linspace(-0.8, 0.8, 9), cmap='coolwarm', alpha=1, extend='both')

    # contour for the streamfunction
    im2  =  ax.contour(lon, lat, sf, levels=np.linspace(-15, 15, 16), alpha=1, colors='k',)
    ax.clabel(im2, fontsize=6, inline=True)

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['///'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='50m', lw=1.5)


    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left', fontsize=12)
    ax.set_title('150 hPa', loc='right', fontsize=12)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im1, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    cb.ax.set_xticks(np.linspace(-0.8, 0.8, 9))
    cb.ax.tick_params(labelsize=7.5)

    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/{}'.format(pic_name))
    #plt.savefig('test.png', dpi=600)

def main():
    # 1. Firstly, calculate difference between two periods for each experiment for the streamfunction
    ncfile   =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/Aerosol_Research_CESM_BTAL_BTALnEU_200hPa_streamfunction_velocity_potential.nc')
    ncfile_v =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_200_ensemble_mean_UVWZ_DIV_and_ttest.nc') 

    p1 = 1901 ; p2 = 1955
    sf_con, sf_p_con = calculate_linear_trend(p1, p2, ncfile, 'btal_sf')
    sf_neu, sf_p_neu = calculate_linear_trend(p1, p2, ncfile, 'btalneu_sf')
    sf_dif, sf_p_dif = calculate_linear_trend_diff(p1, p2, ncfile, 'btal_sf', 'btalneu_sf')
    v_con,  v_p_con  = calculate_linear_trend(p1, p2, ncfile_v, 'btal_v')
    v_neu,  v_p_neu  = calculate_linear_trend(p1, p2, ncfile_v, 'btalneu_v')
    v_dif,  v_p_dif  = calculate_linear_trend_diff(p1, p2, ncfile_v, 'btal_v', 'btalneu_v')

    #sys.exit()
    print(v_p_dif)


    paint_jjas_diff(sf_dif/1e5 * 1e1,  v_dif * 1e2, sf_p_con, "ERL_fig3_v2_CESM_BTAL_streamfunction_meridional_wind_linear_trend_200.pdf", '(a)')
    print("Paint Success")
#    paint_jjas_diff2(sf_diff/1e5, w_diff, None, "ERL_fig3_type2_rp_v_to_w_CESM_BTAL_streamfunction_meridional_wind_period_diff_150.pdf", '(a)')


if __name__ == '__main__':
    main()