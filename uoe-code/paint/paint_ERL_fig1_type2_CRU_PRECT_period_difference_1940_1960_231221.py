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

2024-1-11 update:
Same to paint_ERL_fig1_supplement_GPCC_PRECT_period_difference_1940_1960_231221.py, except for CRU data
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

file_path = '/exports/csce/datastore/geos/users/s2618078/data/download/CRU_precipitation/'
file_name = 'cru_ts4.00.1901.2015.pre.dat.nc'

f0        = xr.open_dataset(file_path + file_name)

lat       = f0.lat.data # -90 to 90
lon       = f0.lon.data
time      = f0.time.data # 1901-01 total 1380 quantities
#print(time)
varname   = 'pre' # mm/month

# ===============================================================

# ================== Calculation for JJA / JJAS precipitation ================

month0 = [6, 7, 8, 9] # JJAS
month1 = [6, 7, 8] # JJA

avg_JJAS = np.zeros((115, len(lat), len(lon)))
avg_JJA  = np.zeros((115, len(lat), len(lon)))

f0_JJAS  = f0.sel(time=f0.time.dt.month.isin(month0)) ; print(len(f0_JJAS.time.data)/len(month0))
f0_JJA   = f0.sel(time=f0.time.dt.month.isin(month1)) ; print(len(f0_JJA.time.data)/len(month1))

for mm in range(115):
    avg_JJAS[mm] = np.average(f0_JJAS[varname].data[mm * len(month0) : (mm * len(month0) + len(month0))], axis=0)
    avg_JJA[mm]  = np.average(f0_JJA[varname].data[mm * len(month1)  : (mm * len(month1) + len(month1))], axis=0)

print('Both JJAS and JJA has been calculated!')

ncfile  =  xr.Dataset(
    {
        "JJAS_PRECT": (["time", "lat", "lon"], avg_JJAS/31),
        "JJA_PRECT":  (["time", "lat", "lon"], avg_JJA/31),
    },
    coords={
        "time": (["time"], np.linspace(1901, 1901 + 114, 115)),
        "lat":  (["lat"],  lat),
        "lon":  (["lon"],  lon),
    },
    )

ncfile["JJAS_PRECT"].attrs['units'] = 'mm day^-1'
ncfile["JJA_PRECT"].attrs['units']  = 'mm day^-1'

ncfile.attrs['description'] = 'Created on 2024-1-8. This is the JJA and JJAS-mean precipitation from CRU data for the period 1901 to 2017'
ncfile.attrs['script'] = 'paint_ERL_fig1_type2_CRU_PRECT_period_difference_1940_1960_231221.py on UOE'
#
out_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/'
ncfile.to_netcdf(out_path + 'Aerosol_Research_CRU_PRECT_JJA_JJAS_average.nc')

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
    



def paint_trend(lat, lon, diff, level, p, title_name, pic_path, pic_name):
    '''
        This function is plot the trend
    '''
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    from matplotlib import projections
    import cartopy.crs as ccrs

    # --- Set the figure ---
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 12), subplot_kw={'projection': proj})

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  65,93,5,35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(70,90,3,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for precipitation trend
    im  =  ax.contourf(lon, lat, diff, levels=level, cmap='bwr_r', alpha=1, extend='both')

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.05], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # Add a rectangle
    ax.add_patch(mpatches.Rectangle(xy=[74, 18], width=12, height=10,
                                facecolor='none', edgecolor='orange', linewidth=2,
                                transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title('(a)', loc='left', fontsize=17.5)
    ax.set_title('CRU', loc='right', fontsize=17.5)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    #cb.ax.tick_params(labelsize=15, rotation=45)

    plt.savefig(pic_path + pic_name)
#

# =================== Calculation for period difference ==========================

periodA_1 = 1900 ; periodA_2 = 1920
periodB_1 = 1940 ; periodB_2 = 1960

data_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/'

f0        = xr.open_dataset(data_path + 'Aerosol_Research_CRU_PRECT_JJA_JJAS_average.nc')

lat       = f0.lat.data ; lon       = f0.lon.data

f0_p1     = f0.sel(time=slice(periodA_1, periodA_2))
f0_p2     = f0.sel(time=slice(periodB_1, periodB_2))

diff_precip_JJAS = np.average(f0_p2['JJAS_PRECT'].data, axis=0) - np.average(f0_p1['JJAS_PRECT'].data, axis=0)
diff_precip_JJA  = np.average(f0_p2['JJA_PRECT'].data, axis=0)  - np.average(f0_p1['JJA_PRECT'].data, axis=0)

# ================== Calculation fot ttest ========================================

# Prepare for the anomaly array
anomaly_periodA = f0_p1['JJAS_PRECT'].data.copy() ; anomaly_periodB = f0_p2['JJAS_PRECT'].data.copy()
p_value = np.ones((len(lat), len(lon)))

for yy in range(len(f0.lat.data)):
    for xx in range(len(f0.lon.data)):
        if np.isnan(f0_p1['JJAS_PRECT'].data[0, yy, xx]):
            continue
        else:
            anomaly_periodA = f0_p1['JJAS_PRECT'].data[:, yy, xx] - np.average(f0['JJAS_PRECT'].data[:, yy, xx])
            anomaly_periodB = f0_p2['JJAS_PRECT'].data[:, yy, xx] - np.average(f0['JJAS_PRECT'].data[:, yy, xx])
            a,b  = stats.ttest_ind(anomaly_periodA, anomaly_periodB, equal_var=False)
            p_value[yy, xx] = b

#print(np.nanmin(p_value))
#print(np.nanmax(p_value))

ncfile  =  xr.Dataset(
    {
       "pavlue_CRU_JJAS_periods": (["lat", "lon"], p_value),
                                        
    },
    coords={
        "lat":  (["lat"],  f0.lat.data),
        "lon":  (["lon"],  f0.lon.data),
    },
    )

ncfile.attrs['description'] = "Created on 2023-12-27. This is the p_value for GPCC precipitation period difference (1900-1920) (1940-1960)"
ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/Aerosol_research_CRU_JJAS_periods_pvalue.nc")

# ================== DONE for Calculation fot ttest ========================================

def main():
    pvalue = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/Aerosol_research_CRU_JJAS_periods_pvalue.nc")
    paint_trend(lat=lat, lon=lon, diff=diff_precip_JJAS, level=np.linspace(-3, 3, 13), p=pvalue["pavlue_CRU_JJAS_periods"], title_name='JJAS', pic_path='/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/', pic_name="ERL_fig1b_Aerosol_Research_CRU_PRECT_JJAS_period_diff_1900_1960.pdf")
#    paint_trend(lat=lat, lon=lon, diff=diff_precip_JJA, level=np.linspace(-3, 3, 13), p=None, title_name='JJA', pic_path='/home/sun/paint/aerosol_research/',    pic_name="Aerosol_Research_GPCC_PRECT_JJA_period_diff_1900_1960.pdf")
##
if __name__ == '__main__':
    main()