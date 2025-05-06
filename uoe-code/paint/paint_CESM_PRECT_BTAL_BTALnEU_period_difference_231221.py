'''
2023-12-21
This script is accompanied with script paint_GPCC_PRECT_period_difference_1940_1960_231221.py on Huaibei
to compare the model simulationwith the observation-based data
'''
import xarray as xr
import numpy as np
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches

sys.path.append("/exports/csce/datastore/geos/users/s2618078/uoe-code/module/")
from module_sun import *

lonmin,lonmax,latmin,latmax  =  60,120,0,40
extent     =  [lonmin,lonmax,latmin,latmax]

#levels = np.array([-1.8, -1.5, -1.2, -0.9, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.2, 1.5, 1.8])
levels = np.linspace(-1., 1., 21)

# ===================== Calculation for JJA and JJAS precipitation period difference ===========================

data_path = "/exports/csce/datastore/geos/users/s2618078/data/analysis_data/"
data_file = "CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc"

data = xr.open_dataset(data_path + data_file)
lat  = data.lat.data
lon  = data.lon.data

#print(data)
data_p1 = data.sel(time=slice(1900, 1920))
data_p2 = data.sel(time=slice(1940, 1960))

prect_BTAL_JJA_DIFF     = np.average(data_p2['PRECT_JJA_BTAL'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTAL'].data, axis=0)
prect_BTALnEU_JJA_DIFF  = np.average(data_p2['PRECT_JJA_BTALnEU'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTALnEU'].data, axis=0)
prect_BTAL_JJAS_DIFF    = np.average(data_p2['PRECT_JJAS_BTAL'].data, axis=0) - np.average(data_p1['PRECT_JJAS_BTAL'].data, axis=0)
prect_BTALnEU_JJAS_DIFF = np.average(data_p2['PRECT_JJAS_BTALnEU'].data, axis=0) - np.average(data_p1['PRECT_JJAS_BTALnEU'].data, axis=0)

def plot_diff_rainfall(diff_data, left_title, right_title, out_path, pic_name):
    '''This function plot the difference in precipitation'''
    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 12), subplot_kw={'projection': proj})

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  60,120,0,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60,120,7,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for precipitation trend
    im  =  ax.contourf(lon, lat, diff_data, levels=levels, cmap='bwr_r', alpha=1, extend='both')

#    # Stippling picture
#    sp  =  ax.contourf(lon, lat, p, levels=[0.1, 1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # --- add patch for key area ---
    ax.add_patch(mpatches.Rectangle(xy=[74, 18], width=12, height=10,
                                facecolor='none', edgecolor='k',
                                transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=17.5)
    ax.set_title(right_title, loc='right', fontsize=17.5)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    #cb.ax.tick_params(labelsize=15, rotation=45)

    plt.savefig(out_path + pic_name)
    

def main():
    out_path = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/'
    #plot_diff_rainfall(diff_data=prect_BTAL_JJA_DIFF, left_title='BTAL', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTAL_JJA_period_difference_1900_1960_231221.pdf")
    plot_diff_rainfall(diff_data=prect_BTAL_JJAS_DIFF, left_title='BTAL', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTAL_JJAS_period_difference_1900_1960_231221.pdf")
    #plot_diff_rainfall(diff_data=prect_BTALnEU_JJA_DIFF,  left_title='BTALnEU', right_title='JJA',  out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJA_period_difference_1900_1960_231221.pdf")
    plot_diff_rainfall(diff_data=prect_BTALnEU_JJAS_DIFF, left_title='BTALnEU', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJAS_period_difference_1900_1960_231221.pdf")


if __name__ == '__main__':
    main()