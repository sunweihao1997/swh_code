'''
2023-12-11
This script modifed from paint_BTAL_change_Indian_JJAS_ts_SST_sp_20231205_update.py, in this script I will only paint TS, which data has been corrected
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



def paint_jjas_diff_BTAL(sst, ncfile):
    lon = ncfile.lon.data
    lat = ncfile.lat.data

    '''This function paint the Diff aerosol JJA'''
    proj       =  ccrs.PlateCarree()
    ax         =  plt.subplot(projection=proj)

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  0,150,-10,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=[0, 150, 0, 80],xticks=np.linspace(0,150,6,dtype=int),yticks=np.linspace(0,80,9,dtype=int),nx=1,ny=1,labelsize=10)

    # contourf for the geopotential height
    im1  =  ax.contourf(lon, lat, sst, levels=np.linspace(-1, 1, 11), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0.1, 1], colors='none', hatches=['...'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='110m', lw=1.1)
    #ax.add_feature(cartopy.feature.LAND, zorder=100, facecolor='white')
    #ax.set_global()

    #ax.set_ylabel("BTAL long-term changes", fontsize=11)

    ax.set_title('1901-1920 to 1941-1960', fontsize=15)
    ax.set_title('CESM', loc='right', fontsize=15)
    ax.set_title('TS', loc='left', fontsize=15)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    #plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/EUI_BTAL_OMEGA_diff_1900_1960.pdf')
    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/BTAL_diff_JJAS_TS.pdf')

def main():


    # === 5. Calculate difference between two given period ===
    file_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'

    periodA_1 = 1900 ; periodA_2 = 1920
    periodB_1 = 1940 ; periodA_2 = 1960

    TS_BTAL     = xr.open_dataset(file_path + "TS_BTAL_ensemble_mean_JJAS_231020.nc")
    TS_BTALnEU  = xr.open_dataset(file_path + "TS_BTALnEU_ensemble_mean_JJAS_231020.nc")

    TS_BTAL_1     = TS_BTAL.sel(time=slice(1900, 1920))    ; TS_BTAL_2     =  TS_BTAL.sel(time=slice(1940, 1960))
    TS_BTALnEU_1  = TS_BTALnEU.sel(time=slice(1900, 1920)) ; TS_BTALnEU_2  =  TS_BTALnEU.sel(time=slice(1940, 1960))

    TS_diff_BTAL    = np.average(TS_BTAL_2['TS_JJAS'].data, axis=0)    - np.average(TS_BTAL_1['TS_JJAS'].data, axis=0)
    TS_diff_BTALnEU = np.average(TS_BTALnEU_2['TS_JJAS'].data, axis=0) - np.average(TS_BTALnEU_1['TS_JJAS'].data, axis=0)

    TS_diff         = TS_diff_BTAL - TS_diff_BTALnEU # Influence of the EU emission

    paint_jjas_diff_BTAL(TS_diff_BTAL, TS_BTAL)




if __name__ == '__main__':
    main()