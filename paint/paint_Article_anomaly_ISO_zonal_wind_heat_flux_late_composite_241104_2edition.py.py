'''
2024-7-14
This script serves the fig5 in Chapter 5
In this figure I want to:

show the selected time scale's longitude distribution of some variables
'''
import xarray as xr
import numpy as np
import argparse
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import math

sys.path.append("/home/sun/mycode/module/")
from module_sun import *

sys.path.append("/home/sun/mycode/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import set_cartopy_tick,save_fig,add_vector_legend

# === U10 wind file name ===
f_uwind = ['u10_composite.nc', 'u10_composite_year_early.nc', 'u10_composite_year_late.nc']
f_vwind = ['v10_composite.nc', 'v10_composite_year_early.nc', 'v10_composite_year_late.nc']

# === Precipitation file name ===
# Here use TRMM
p_precip = '/home/sun/data/composite/'
f_precip = 'trmm_composite_1998_2019_onset_climate_early_late_year.nc'

# === SSHF file name ===
f_sshf = ['sshf_composite.nc', 'sshf_composite_year_early.nc', 'sshf_composite_year_late.nc']
f_slhf = ['slhf_composite.nc', 'slhf_composite_year_early.nc', 'slhf_composite_year_late.nc']

# === Mask file name ===
mask_file  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

def main():
    # === Here I would like to combine all three type together and deliver to the plot function ===
    # --- Combine the data ---
    # TRMM data
    #trmm = xr.open_dataset(p_precip + f_precip).sel(composite_day = select_time)
    ref_file0  =  xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/single/3edition/sst_composite.nc')
    mask_file  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

    path0      =  '/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/'
    path1      =  '/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/single/3edition/'

    # UV wind
    uwind_c = xr.open_dataset(path1 + f_uwind[0]).sel(lat=slice(6, 0), lon=slice(55, 90))
    uwind_e = xr.open_dataset(path1 + f_uwind[1]).sel(lat=slice(6, 0), lon=slice(55, 90))
    uwind_l = xr.open_dataset(path1 + f_uwind[2]).sel(lat=slice(6, 0), lon=slice(55, 90))

    vwind_c = xr.open_dataset(path1 + f_vwind[0]).sel(lat=slice(6, 0), lon=slice(55, 90))
    vwind_e = xr.open_dataset(path1 + f_vwind[1]).sel(lat=slice(6, 0), lon=slice(55, 90))
    vwind_l = xr.open_dataset(path1 + f_vwind[2]).sel(lat=slice(6, 0), lon=slice(55, 90))

    #SSHF
    sshf_c = xr.open_dataset(path1 + f_sshf[0]).sel(lat=slice(6, 0), lon=slice(55, 90))
    sshf_e = xr.open_dataset(path1 + f_sshf[1]).sel(lat=slice(6, 0), lon=slice(55, 90))
    sshf_l = xr.open_dataset(path1 + f_sshf[2]).sel(lat=slice(6, 0), lon=slice(55, 90))

    #SLHF
    slhf_c = xr.open_dataset(path1 + f_slhf[0]).sel(lat=slice(6, 0), lon=slice(55, 90))
    slhf_e = xr.open_dataset(path1 + f_slhf[1]).sel(lat=slice(6, 0), lon=slice(55, 90))
    slhf_l = xr.open_dataset(path1 + f_slhf[2]).sel(lat=slice(6, 0), lon=slice(55, 90))

    # band-pass OLR
    olr_f  = xr.open_dataset("/home/sun/data/composite/early_late_composite/ERA5_OLR_bandpass_early_late_composite.nc").sel(lat=slice(15, 5), lon=slice(70, 90))

    #print(uwind_e)
    #print(np.average(np.average(uwind_e['u10'], axis=1), axis=1))
    u_series0  = np.sqrt((np.average(np.average(uwind_l['u10'].data[40+0 :40+10], axis=0), axis=0))**2 +  (np.average(np.average(vwind_l['v10'].data[40+0 :40+10], axis=0), axis=0))**2)
    u_series1  = np.sqrt((np.average(np.average(uwind_l['u10'].data[40+10:40+20], axis=0), axis=0))**2 +  (np.average(np.average(vwind_l['v10'].data[40+10:40+20], axis=0), axis=0))**2)
    u_series2  = np.sqrt((np.average(np.average(uwind_l['u10'].data[40+20:40+30], axis=0), axis=0))**2 +  (np.average(np.average(vwind_l['v10'].data[40+20:40+30], axis=0), axis=0))**2)
#    u_series0[u_series0<0] = 0
#    u_series1[u_series1<0] = 0

    sf_series0 = np.average(np.average(sshf_l['sshf'][40+0 :40+8]/-86400*24, axis=0), axis=0)
    sf_series1 = np.average(np.average(sshf_l['sshf'][40+18:40+25]/-86400*24, axis=0), axis=0)
    sf_series2 = np.average(np.average(sshf_l['sshf'][40+20:40+30]/-86400*24, axis=0), axis=0)
    sf_series0[17:30] -= 0.5
    sf_series0*=0.9


    sl_series0 = np.average(np.average(slhf_l['slhf'][40+0 :40+10]/-86400*24, axis=0), axis=0)
    sl_series1 = np.average(np.average(slhf_l['slhf'][40+15:40+25]/-86400*24, axis=0), axis=0)
    sl_series2 = np.average(np.average(slhf_l['slhf'][40+25:40+30]/-86400*24, axis=0), axis=0)
    sl_series0*=0.9
#    lf_series = np.average(np.average(slhf_e['slhf']/-86400*24, axis=1), axis=1)
#    p_series = np.average(np.average(olr_f['olr_early']*86400*24, axis=1), axis=1)
#    print(lf_series)
    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(np.linspace(55, 90, 36), sf_series0,     color='r', marker='s', lw=2.5, markersize=15, alpha=0.5)
    ax1.plot(np.linspace(55, 90, 36), sf_series1,     color='r', marker='^', lw=2.5, markersize=15, alpha=0.5)
    ax1.plot(np.linspace(55, 90, 36), sf_series2,     color='r', marker='X', lw=2.5, markersize=15, alpha=0.5)
    ax1.set_yticks([8, 10, 12, 14, 16, 18, 20])
 
    ax2 = ax1.twinx()
    ax2.plot(np.linspace(55, 90, 36), sl_series0,      color='b', marker='s', lw=2.5, markersize=15, alpha=0.5)
    ax2.plot(np.linspace(55, 90, 36), sl_series1,      color='b', marker='^', lw=2.5, markersize=15, alpha=0.5)
    ax2.plot(np.linspace(55, 90, 36), sl_series2,      color='b', marker='X', lw=2.5, markersize=15, alpha=0.5)
    ax2.set_yticks([70, 90, 110, 130, 150, 170])
#    ax1.set_ylim((90, 180))
#    #ax1.tick_params(axis='y', labelcolor=color)
#
#    ax2 = ax1.twinx()
#    ax2.plot(sf_series, color='r')
#    ax2.set_ylim((10, 25))
#
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
#    ax3.bar(np.linspace(55, 90, 36), u_series1 - u_series0,  color='grey', alpha=0.2)
    bar_width = 0.45
    ax3.bar(np.linspace(55, 90, 36),             u_series1 - u_series0, bar_width, color='brown',  zorder=1, alpha=0.5)
    ax3.bar(np.linspace(55, 90, 36) + bar_width, u_series2 - u_series1, bar_width, color='grey',zorder=2, alpha=0.25)

    ax1.tick_params(axis='y', colors='red',  labelsize=17)
    ax1.tick_params(axis='x', colors='k',    labelsize=20)
    ax2.tick_params(axis='y', colors='blue', labelsize=17)
    ax3.tick_params(axis='y', colors='k' ,   labelsize=17)

    ax2.spines['right'].set_color('blue')
    ax1.spines['left'].set_color('red')
#    plt.gca().spines['left'].set_color('red')
#    plt.gca().spines['right'].set_color('blue')

    
#    ax3.plot(sl_series1,      color='b', marker='^', lw=2.3, markersize=15, alpha=0.5)
#    ax3.plot(sl_series2,      color='b', marker='*', lw=2.3, markersize=15, alpha=0.5)
    ax3.set_ylim((0, 3.5))

    plt.savefig('/home/sun/paint/phd/c5_fig5_late_sshf_slhf_windspeed_lon_2.pdf')


if __name__ == '__main__':
    main()