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
from scipy import stats
from scipy.ndimage import gaussian_filter

sys.path.append("/exports/csce/datastore/geos/users/s2618078/uoe-code/module/")
from module_sun import *

lonmin,lonmax,latmin,latmax  =  60,120,0,40
extent     =  [lonmin,lonmax,latmin,latmax]

mask_file  =  xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/land-sea_mask.nc")

#levels = np.array([-1.8, -1.5, -1.2, -0.9, -0.6, -0.5, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.9, 1.2, 1.5, 1.8])
#levels = np.linspace(-1., 1., 21)
levels = np.linspace(-0.5, 0.5, 11)


# ===================== Calculation for JJA and JJAS precipitation period difference ===========================

data_path = "/exports/csce/datastore/geos/users/s2618078/data/analysis_data/"
data_file = "CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc"

data = xr.open_dataset(data_path + data_file)
lat  = data.lat.data
lon  = data.lon.data

mask_file_interp = mask_file.interp(latitude=lat, longitude=lon, method='slinear')
#print(mask_file_interp.lsm.data[0])

#print(data)
data_p1 = data.sel(time=slice(1901, 1920))
data_p2 = data.sel(time=slice(1940, 1960))
data_p3 = data.sel(time=slice(1945, 1960))

#prect_BTAL_JJA_DIFF     = np.average(data_p2['PRECT_JJA_BTAL'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTAL'].data, axis=0)
#prect_BTALnEU_JJA_DIFF  = np.average(data_p2['PRECT_JJA_BTALnEU'].data, axis=0) - np.average(data_p1['PRECT_JJA_BTALnEU'].data, axis=0)
prect_BTAL_JJAS_DIFF    = np.average(data_p2['PRECT_JJAS_BTAL'].data, axis=0) - np.average(data_p1['PRECT_JJAS_BTAL'].data, axis=0)
prect_BTALnEU_JJAS_DIFF = np.average(data_p2['PRECT_JJAS_BTALnEU'].data, axis=0) - np.average(data_p1['PRECT_JJAS_BTALnEU'].data, axis=0)

# Mask the data over ocean
#prect_BTAL_JJAS_DIFF[mask_file_interp['lsm'].data[0]==0] = np.nan
#prect_BTALnEU_JJAS_DIFF[mask_file_interp['lsm'].data[0]==0] = np.nan

# ========================= Calculate ttest ===============================

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
            p_array[yy, xx] = p_value

    return p_array

def plot_diff_rainfall(diff_data, left_title, right_title, out_path, pic_name, p):
    '''This function plot the difference in precipitation'''
    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(20, 14), subplot_kw={'projection': proj})

#    # --- Set range ---
#    lonmin,lonmax,latmin,latmax  =  65,93,5,35
#    extent     =  [lonmin,lonmax,latmin,latmax]
#
#    # --- Tick setting ---
#    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(70,90,3,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    #lonmin,lonmax,latmin,latmax  =  65,93,5,35
    lonmin,lonmax,latmin,latmax  =  60,125,5,35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(10,60,6,dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for precipitation trend
    im  =  ax.contourf(lon, lat, diff_data, levels=levels, cmap='coolwarm_r', alpha=1, extend='both')

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['.'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # --- add patch for key area ---
    ax.add_patch(mpatches.Rectangle(xy=[72, 20], width=12, height=7.5,linestyle='--',
                                facecolor='none', edgecolor='grey', linewidth=3.5,
                                transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=22.5)
    ax.set_title(right_title, loc='right', fontsize=22.5)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=15,)

    plt.savefig(out_path + pic_name)
    
# ================== Calculation fot ttest ========================================

# Prepare for the anomaly array
#anomaly_periodA = f0_p1['JJAS_PRECT'].data.copy() ; anomaly_periodB = f0_p2['JJAS_PRECT'].data.copy()

p_value_BTAL         = np.ones((len(lat), len(lon)))
p_value_BTALnEU      = np.ones((len(lat), len(lon)))
p_value_BTAL_BTALnEU = np.ones((len(lat), len(lon)))

for yy in range(len(lat)):
    for xx in range(len(lon)):
        anomaly_periodA_BTAL = data_p1['PRECT_JJAS_BTAL'].data[:, yy, xx] - np.average(data['PRECT_JJAS_BTAL'].data[:, yy, xx])
        anomaly_periodB_BTAL = data_p2['PRECT_JJAS_BTAL'].data[:, yy, xx] - np.average(data['PRECT_JJAS_BTAL'].data[:, yy, xx])
        a,b  = stats.ttest_ind(anomaly_periodA_BTAL, anomaly_periodB_BTAL, equal_var=False)
        p_value_BTAL[yy, xx] = b

        anomaly_periodA_BTALnEU = data_p1['PRECT_JJAS_BTALnEU'].data[:, yy, xx] - np.average(data['PRECT_JJAS_BTALnEU'].data[:, yy, xx])
        anomaly_periodB_BTALnEU = data_p2['PRECT_JJAS_BTALnEU'].data[:, yy, xx] - np.average(data['PRECT_JJAS_BTALnEU'].data[:, yy, xx])
        a,b  = stats.ttest_ind(anomaly_periodA_BTALnEU, anomaly_periodB_BTALnEU, equal_var=False)
        p_value_BTALnEU[yy, xx] = b

        anomaly_periodC_BTAL    = data_p3['PRECT_JJAS_BTAL'].data[:, yy, xx] - np.average(data['PRECT_JJAS_BTAL'].data[:, yy, xx])
        anomaly_periodC_BTALnEU = data_p3['PRECT_JJAS_BTALnEU'].data[:, yy, xx] - np.average(data['PRECT_JJAS_BTALnEU'].data[:, yy, xx])
        a,b  = stats.ttest_ind(anomaly_periodC_BTAL, anomaly_periodC_BTALnEU, equal_var=False)
        p_value_BTAL_BTALnEU[yy, xx] = b

p_value_BTAL[mask_file_interp['lsm'].data[0]<=0]    = np.nan
p_value_BTALnEU[mask_file_interp['lsm'].data[0]<=0] = np.nan
p_value_BTAL_BTALnEU[mask_file_interp['lsm'].data[0]<=0] = np.nan

def main():
    out_path = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/'
    #plot_diff_rainfall(diff_data=prect_BTAL_JJA_DIFF, left_title='BTAL', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTAL_JJA_period_difference_1900_1960_231221.pdf")
    #plot_diff_rainfall(diff_data=prect_BTALnEU_JJA_DIFF,  left_title='BTALnEU', right_title='JJA',  out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJA_period_difference_1900_1960_231221.pdf")
    plot_diff_rainfall(diff_data=gaussian_filter(prect_BTAL_JJAS_DIFF, sigma=0.8), left_title='(c)', right_title='CESM_ALL (JJAS)', out_path=out_path, pic_name="ERL_fig1c_CESM_prect_BTAL_JJAS_period_difference_1900_1960_231221.pdf", p=p_value_BTAL)
    plot_diff_rainfall(diff_data=prect_BTALnEU_JJAS_DIFF, left_title='(b)', right_title='CESM_noEU (JJAS)', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJAS_period_difference_1900_1960_231221.pdf", p=p_value_BTALnEU)
    plot_diff_rainfall(diff_data=gaussian_filter((prect_BTAL_JJAS_DIFF - prect_BTALnEU_JJAS_DIFF), sigma=1), left_title='(d)', right_title='CESM_ALL - CESM_noEU (JJAS)', out_path=out_path, pic_name="ERL_fig1d_CESM_prect_BTAL_sub_BTALnEU_JJAS_period_difference_1900_1960_231227.pdf", p=p_value_BTAL_BTALnEU)


if __name__ == '__main__':
    main()