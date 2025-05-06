'''
2023-10-6
This script is to calculate the difference between (1901 to 1920) and (1941 to 1960) period
to show the change in SLP trend over the Indian continent
'''
import xarray as xr
import numpy as np
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys

sys.path.append("/exports/csce/datastore/geos/users/s2618078/uoe-code/module/")
from module_sun import *

lonmin,lonmax,latmin,latmax  =  40,115,-10,40
extent     =  [lonmin,lonmax,latmin,latmax]

#levels = np.array([-1.8, -1.5, -1.2, -0.9, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.2, 1.5, 1.8])
levels = np.linspace(-100, 100, 21)

def cal_diff_select_period(start1, end1, start2, end2, path, file, variable):
    '''This function calculate variable difference in the given period'''
    file0 = xr.open_dataset(path + file)

    print(np.average(file0[variable].data[start1 : end1]))
    diff  = np.average(file0[variable].data[start1 : end1], axis=0) - np.average(file0[variable].data[start2 : end2], axis=0)

    return diff

def plot_diff_rainfall(var_all, var_EU, var_diff, extent):
    '''This function plot the difference in precipitation'''
    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    ref_file = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_precipitation_ensemble_mean_231005.nc')

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig1    =  plt.figure(figsize=(20,35))
    spec1   =  fig1.add_gridspec(nrows=3,ncols=1)

    # ------------ First. Paint the All forcing picture ------------
    ax1 = fig1.add_subplot(spec1[0, 0], projection=proj)

    # Tick setting
    set_cartopy_tick(ax=ax1,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=25)

    # Equator line
    ax1.plot([40,120],[0,0],'k--')

    # Set ylabel name
    ax1.set_ylabel('ALL_FORCING', fontsize=25)

    # Set title
    ax1.set_title('1941to1960 - 1901to1921', fontsize=25)

    # Shading for precipitation
    im2  =  ax1.contourf(ref_file['lon'].data, ref_file['lat'].data, var_all * -1, levels, cmap='bwr_r', alpha=1, extend='both')

    # Coast Line
    ax1.coastlines(resolution='110m', lw=1.75)

    # ------------ Second. Paint the noEU forcing picture ------------
    ax2 = fig1.add_subplot(spec1[1, 0], projection=proj)

    # Tick setting
    set_cartopy_tick(ax=ax2,extent=extent,xticks=np.linspace(50,115,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=25)

    # Equator line
    ax2.plot([40,120],[0,0],'k--')

    # Shading for precipitation
    im2  =  ax2.contourf(ref_file['lon'].data, ref_file['lat'].data, var_EU * -1, levels, cmap='bwr_r', alpha=1, extend='both')

    # Coast Line
    ax2.coastlines(resolution='110m', lw=1.75)

    # Set ylabel name
    ax2.set_ylabel('No_EU', fontsize=25)

    # Set title
    ax2.set_title('1941to1960 - 1901to1921', fontsize=25)

    # ------------ Third. Paint the All forcing subtract noEU picture ------------
    ax3 = fig1.add_subplot(spec1[2, 0], projection=proj)

    # Tick setting
    set_cartopy_tick(ax=ax3,extent=extent,xticks=np.linspace(50,115,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=25)

    # Equator line
    ax3.plot([40,120],[0,0],'k--')

    # Shading for precipitation
    im3  =  ax3.contourf(ref_file['lon'].data, ref_file['lat'].data, var_diff * -1, levels, cmap='bwr_r', alpha=1, extend='both')

    # Coast Line
    ax3.coastlines(resolution='110m', lw=1.75)

    # Set ylabel name
    ax3.set_ylabel('ALL_FORCING - No_EU', fontsize=25)

    # Set title
    ax3.set_title('1941to1960 - 1901to1921', fontsize=25)

    # ========= add colorbar =================
    fig1.subplots_adjust(top=0.8) 
    cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig1.colorbar(im3, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=15, rotation=45)

    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/slp1901to1920_1941to1960_difference_slp_Indian_continent.pdf', dpi=500)

    


def main():
    path0 = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    all_forcing = cal_diff_select_period(start1=50, end1=70, start2=90, end2=110, path=path0, file='BTAL_SLP_jja_mean_231005.nc', variable='SLP_JJA')
    print(np.average(all_forcing))
    noEU_forcing= cal_diff_select_period(start1=50, end1=70, start2=90, end2=110, path=path0, file='noEU_SLP_jja_mean_231005.nc', variable='SLP_JJA')
    diff        = all_forcing - noEU_forcing

    plot_diff_rainfall(var_all=all_forcing, var_EU=noEU_forcing, var_diff=diff, extent=extent)

if __name__ == '__main__':
    main()
