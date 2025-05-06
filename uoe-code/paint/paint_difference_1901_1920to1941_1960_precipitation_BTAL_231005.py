'''
2023-10-5
This script is to calculate the difference between (1901 to 1920) and (1941 to 1960) period
to show the change in precipitation trend over the Indian continent

2023-12-21 Notice: This script has been dumed, please move to paint_CESM_PRECT_BTAL_BTALnEU_period_difference_231221.py
'''
import xarray as xr
import numpy as np
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/sun/uoe-code/module/")
from module_sun import *

lonmin,lonmax,latmin,latmax  =  40,130,0,40
extent     =  [lonmin,lonmax,latmin,latmax]

#cmap  =  create_ncl_colormap("/exports/csce/datastore/geos/users/s2618078/data/ncl_colormap/MPL_coolwarm.txt",22)
#print(cmap)

# 2.1 Set the colormap for precipitation
#from matplotlib import cm
#from matplotlib.colors import ListedColormap
#viridis = cm.get_cmap('BrBG', 22)
#newcolors = viridis(np.array([0, 0.05, 0.1, 0.15, 0.2, 0.225, 0.275, 0.3, 0.35, 0.4, 0.6, 0.65, 0.7, 0.725, 0.775, 0.8, 0.85, 0.9, 0.95, 1]))
#newcmp = ListedColormap(newcolors)
##newcmp.set_under('white')
##newcmp.set_over('#145DA0')

levels = np.array([-1.8, -1.5, -1.2, -0.9, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.2, 1.5, 1.8])

data_path = "/home/sun/data/download_data/data/analysis_data/"
data_file = "CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc"

data = xr.open_dataset(data_path + data_file)
lat  = data.lat.data
lon  = data.lon.data

def cal_diff_select_period(start1, end1, start2, end2, path, file, variable):
    '''This function calculate variable difference in the given period'''
    file0 = xr.open_dataset(path + file)

    diff  = np.average(file0[variable].data[start1 : end1], axis=0) - np.average(file0[variable].data[start2 : end2], axis=0)

    return diff

def plot_diff_rainfall(var_all, var_EU, var_diff, extent):
    '''This function plot the difference in precipitation'''
    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    ref_file = xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_precipitation_ensemble_mean_231005.nc')

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
    im2  =  ax1.contourf(ref_file['lon'].data, ref_file['lat'].data, var_all * 1000 * 86400 * -1, levels, cmap='bwr_r', alpha=1, extend='both')

    # Coast Line
    ax1.coastlines(resolution='110m', lw=1.75)

    # ------------ Second. Paint the noEU forcing picture ------------
    ax2 = fig1.add_subplot(spec1[1, 0], projection=proj)

    # Tick setting
    set_cartopy_tick(ax=ax2,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=25)

    # Equator line
    ax2.plot([40,120],[0,0],'k--')

    # Shading for precipitation
    im2  =  ax2.contourf(ref_file['lon'].data, ref_file['lat'].data, var_EU * 1000 * 86400 * -1, levels, cmap='bwr_r', alpha=1, extend='both')

    # Coast Line
    ax2.coastlines(resolution='110m', lw=1.75)

    # Set ylabel name
    ax2.set_ylabel('No_EU', fontsize=25)

    # Set title
    ax2.set_title('1941to1960 - 1901to1921', fontsize=25)

    # ------------ Third. Paint the All forcing subtract noEU picture ------------
    ax3 = fig1.add_subplot(spec1[2, 0], projection=proj)

    # Tick setting
    set_cartopy_tick(ax=ax3,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=25)

    # Equator line
    ax3.plot([40,120],[0,0],'k--')

    # Shading for precipitation
    im3  =  ax3.contourf(ref_file['lon'].data, ref_file['lat'].data, var_diff * 1000 * 86400 * -1, levels, cmap='bwr_r', alpha=1, extend='both')

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

    plt.savefig('/home/sun/paint/ERL/1901to1920_1941to1960_difference_precipitation_Indian_continent.pdf', dpi=500)

    


def main():
    path0 = '/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    all_forcing = cal_diff_select_period(start1=50, end1=70, start2=90, end2=110, path=path0, file='BTAL_precipitation_jja_mean_231005.nc', variable='PRECT_JJA')
    noEU_forcing= cal_diff_select_period(start1=50, end1=70, start2=90, end2=110, path=path0, file='noEU_precipitation_jja_mean_231005.nc', variable='PRECT_JJA')
    diff        = all_forcing - noEU_forcing

    plot_diff_rainfall(var_all=all_forcing, var_EU=noEU_forcing, var_diff=diff, extent=extent)

if __name__ == '__main__':
    main()
