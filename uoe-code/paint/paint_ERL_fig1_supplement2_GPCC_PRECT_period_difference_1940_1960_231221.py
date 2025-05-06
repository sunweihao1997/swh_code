'''
2023-11-20
This script is to paint the trend in precipitation for the period 1900 to 1960

2023-11-22 Update:
1. Change the method for stippling, not use M-K trend test. From the advice of Massimo, when the sign among the CESM ensemble is similar, then stipple it
2. Add result of the noEU experiment

2023-12-21 update:
modified from paint_GPCC_CESM_Indian_precipitation_trend_1900_1960_231120.py
From discussion with Massimo, the point includes:
1. enlarge the scale to see the quality of simulation over China, and change to JJA
2. instead of using trend, here used period difference

2024-1-8 update:
move to the stream server, some changes are needed
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import matplotlib.patches as mpatches
from scipy import stats

module_path = '/exports/csce/datastore/geos/users/s2618078/uoe-code/module/'
sys.path.append(module_path)
from module_sun import *

# =================== File Location =============================

file_path = '/exports/csce/datastore/geos/users/s2618078/data/download/GPCC_NCEP_prect/'
file_name = 'precip.mon.total.1x1.v2020.nc'

f0        = xr.open_dataset(file_path + file_name)

lat       = f0.lat.data # 90 to -90
lon       = f0.lon.data
time      = f0.time.data # 1891 -01 -01
#print(time)
varname   = 'precip'

# ===============================================================

# ================== Calculation for JJA / JJAS precipitation ================

month0 = [6, 7, 8, 9] # JJAS
month1 = [6, 7, 8] # JJA

avg_JJAS = np.zeros((129, 4, len(lat), len(lon)))

j = 0
for m0 in month0:
    f0_single_month = f0.sel(time=f0.time.dt.month.isin([m0]))

    avg_JJAS[:, j] = f0_single_month[varname].data/31

    j += 1


print(np.nanmean(avg_JJAS))

ncfile  =  xr.Dataset(
    {
        "JJAS_PRECT": (["year", "month", "lat", "lon"], avg_JJAS),
    },
    coords={
        "year": (["year"], np.linspace(1891, 1891 + 128, 129)),
        "month":(["month"],month0),
        "lat":  (["lat"],  lat),
        "lon":  (["lon"],  lon),
    },
    )

ncfile["JJAS_PRECT"].attrs['units'] = 'mm day^-1'


ncfile.attrs['description'] = 'Created on 2024-5-3.'
ncfile.attrs['script'] = 'paint_ERL_fig1_supplement2_GPCC_PRECT_period_difference_1940_1960_231221.py on UOE'
#
out_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/'
ncfile.to_netcdf(out_path + 'Aerosol_Research_GPCC_PRECT_singlemonth_JJAS_average.nc')

# ================================================================================



#print(diff_precip_JJAS.shape) # (360, 720)

# ================================================================================

def cal_ttest(data1, data2):
    '''
        This function calculate difference using ttest
        data1 and data2 should be anomaly value compared with climatology
    '''
    import scipy.stats as stats

    # claim the array to save the p value
    p_array = np.zeros((data1.shape[1], data1.shape[2])) # it should be (lat, lon)

    # Compare on each point
    y_num = data1.shape[1]  ;  x_num = data1.shape[2]
    for yy in range(y_num):
        for xx in range(x_num):
            t_value, p_value = stats.ttest_ind(data1[:, yy, xx], data2[:, yy, xx])
            print(p_value)
            p_array[yy, xx] = p_value

    return p_array
    



def plot_diff_rainfall(diff_data, levels, left_title, right_title, out_path, pic_name, p):
    '''This function plot the difference in precipitation'''
    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 14), subplot_kw={'projection': proj})

#    # --- Set range ---
#    lonmin,lonmax,latmin,latmax  =  65,93,5,35
#    extent     =  [lonmin,lonmax,latmin,latmax]
#
#    # --- Tick setting ---
#    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(70,90,3,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  65,93,5,35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(70,90,3,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for precipitation trend
    im  =  ax.contourf(lon, lat, diff_data, levels=levels, cmap='bwr_r', alpha=1, extend='both')

#    # Stippling picture
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # --- add patch for key area ---
    #ax.add_patch(mpatches.Rectangle(xy=[74, 18], width=12, height=10,linewidth=1.5,
    #                            facecolor='none', edgecolor='orange',
    #                            transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=17.5)
    ax.set_title(right_title, loc='right', fontsize=17.5)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=15,)

    plt.savefig(out_path + pic_name)
#

# =================== Calculation for period difference ==========================

periodA_1 = 1900 ; periodA_2 = 1920
periodB_1 = 1940 ; periodB_2 = 1960

data_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/'

f0        = xr.open_dataset(data_path + 'Aerosol_Research_GPCC_PRECT_singlemonth_JJAS_average.nc')
print(f0)

lat       = f0.lat.data ; lon       = f0.lon.data

f0_p1     = f0.sel(year=slice(periodA_1, periodA_2))
f0_p2     = f0.sel(year=slice(periodB_1, periodB_2))

print(f0_p2['JJAS_PRECT'].data.shape)

diff_precip_JJAS = np.nanmean(f0_p2['JJAS_PRECT'].data, axis=0) - np.nanmean(f0_p1['JJAS_PRECT'].data, axis=0)

print(np.nanmax(diff_precip_JJAS))

# ================== Calculation fot ttest ========================================

# Prepare for the anomaly array
#anomaly_periodA = f0_p1['JJAS_PRECT'].data.copy() ; anomaly_periodB = f0_p2['JJAS_PRECT'].data.copy()
#p_value = np.ones((len(lat), len(lon)))
#
#for yy in range(len(f0.lat.data)):
#    for xx in range(len(f0.lon.data)):
#        if np.isnan(f0_p1['JJAS_PRECT'].data[0, yy, xx]):
#            continue
#        else:
#            anomaly_periodA = f0_p1['JJAS_PRECT'].data[:, yy, xx] - np.average(f0['JJAS_PRECT'].data[:, yy, xx])
#            anomaly_periodB = f0_p2['JJAS_PRECT'].data[:, yy, xx] - np.average(f0['JJAS_PRECT'].data[:, yy, xx])
#            a,b  = stats.ttest_ind(anomaly_periodA, anomaly_periodB, equal_var=False)
#            p_value[yy, xx] = b

#print(np.nanmin(p_value))
#print(np.nanmax(p_value))

#ncfile  =  xr.Dataset(
#    {
#       "pavlue_GPCC_JJAS_periods": (["lat", "lon"], p_value),
#                                        
#    },
#    coords={
#        "lat":  (["lat"],  f0.lat.data),
#        "lon":  (["lon"],  f0.lon.data),
#    },
#    )
#
#ncfile.attrs['description'] = "Created on 2023-12-27. This is the p_value for GPCC precipitation period difference (1900-1920) (1940-1960)"
#ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/Aerosol_research_GPCC_JJAS_periods_pvalue.nc")

# ================== DONE for Calculation fot ttest ========================================

def main():
#    pvalue = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/Aerosol_research_GPCC_JJAS_periods_pvalue.nc")
    plot_diff_rainfall(diff_data=diff_precip_JJAS[0], levels=np.linspace(-3, 3, 13), p=None, left_title='June',      right_title='GPCC (1941-1960 minus 1901-1920)',out_path='/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/', pic_name="ERL_supplement_Aerosol_Research_GPCC_PRECT_June_period_diff_1900_1960.pdf")
    plot_diff_rainfall(diff_data=diff_precip_JJAS[1], levels=np.linspace(-3, 3, 13), p=None, left_title='July',      right_title='GPCC (1941-1960 minus 1901-1920)',out_path='/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/', pic_name="ERL_supplement_Aerosol_Research_GPCC_PRECT_July_period_diff_1900_1960.pdf")
    plot_diff_rainfall(diff_data=diff_precip_JJAS[2], levels=np.linspace(-3, 3, 13), p=None, left_title='August',    right_title='GPCC (1941-1960 minus 1901-1920)',out_path='/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/', pic_name="ERL_supplement_Aerosol_Research_GPCC_PRECT_Augu_period_diff_1900_1960.pdf")
    plot_diff_rainfall(diff_data=diff_precip_JJAS[3], levels=np.linspace(-3, 3, 13), p=None, left_title='September', right_title='GPCC (1941-1960 minus 1901-1920)',out_path='/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/', pic_name="ERL_supplement_Aerosol_Research_GPCC_PRECT_Sept_period_diff_1900_1960.pdf")

#    paint_trend(lat=lat, lon=lon, diff=diff_precip_JJA, level=np.linspace(-3, 3, 13), p=None, title_name='JJA', pic_path='/home/sun/paint/aerosol_research/',    pic_name="Aerosol_Research_GPCC_PRECT_JJA_period_diff_1900_1960.pdf")
##
if __name__ == '__main__':
    main()