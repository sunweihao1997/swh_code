'''
2024-9-19
This script serves for the Figure S1 in ERL
The purpose is to evaluate the simulation of CESM on summer monsoon

variables:
uv 850 ; Precip for JJA
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
from scipy import stats
#import cmasher as cmr
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.colors import ListedColormap

sys.path.append('/home/sun/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# =================== Calculating Part ========================
# 1.1 calculate the JJA mean for ERA5
# !!Note!! Due to ERA5 data not saved on Huaibei Server, this data was generated on ubuntu by script:
# cal_ERL_figs1_model_evaluation_850wind_pr_240919.py
fera5 = xr.open_dataset("/home/sun/data/process/analysis/ERA5/ERA5_monthly_JJA_wind_precip.nc")

# 1.2 calculate the JJA mean for CESM
fcesm_pr = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc")
fcesm_u  = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/U_BTAL_ensemble_mean_JJA_231019.nc")
fcesm_v  = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/V_BTAL_ensemble_mean_JJA_231019.nc")
fcesm    = xr.merge([fcesm_pr, fcesm_u, fcesm_v])
fcesm_sel= fcesm.sel(time=slice(1980, 2005))

# 1.2.1 calculate mean for selected period
fcesm_men= fcesm_sel.mean(dim="time", skipna=True)
print(fcesm_men)
#sys.exit()

# 1.3 Interpolate ERA5 data to the CESM grid
fera5_interp = fera5.interp(lat=fcesm.lat.data, lon=fcesm.lon.data)
#print(fcesm_men)

# !================== End of calculating =======================!

# ================== Plot Part ================================
# Vector: Wind ; Shading: Precipitation
# Set the figure
proj    =  ccrs.PlateCarree()
fig, ax =  plt.subplots(3, 1, figsize=(15, 25), subplot_kw={'projection': proj})

# coordinate information
lat = fcesm_sel.lat.data ; lon = fcesm_sel.lon.data
levels = np.arange(2, 35, 5)

# --- Set range ---
lonmin,lonmax,latmin,latmax  =  60,125,5,35
extent     =  [lonmin,lonmax,latmin,latmax]

# ------ ERA5 Poltting ---------
# --- Tick setting ---
set_cartopy_tick(ax=ax[0],extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(10,60,6,dtype=int),nx=1,ny=1,labelsize=25)

im1  =  ax[0].contourf(lon, lat, gaussian_filter((fera5_interp['tp'].data * 1e3 ), sigma=0.5), levels=levels, cmap='Blues', alpha=1, extend='max')

# Vectors for Wind difference
q  =  ax[0].quiver(lon, lat, fera5_interp.sel(level=850)['u'].data, fera5_interp.sel(level=850)['v'].data, 
                    regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                    scale_units='xy', scale=1.95,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                    units='xy', width=0.35,              # width控制粗细
                    transform=proj,
                    color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

add_vector_legend(ax=ax[0], q=q, speed=5)

# --- Coast Line ---
ax[0].coastlines(resolution='110m', lw=1.25)

# ------ CESM Poltting ---------
# --- Tick setting ---
set_cartopy_tick(ax=ax[1],extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(10,60,6,dtype=int),nx=1,ny=1,labelsize=25)

im2  =  ax[1].contourf(lon, lat, gaussian_filter((fcesm_men['PRECT_JJA_BTAL'].data), sigma=0.5)*1.1, levels=levels, cmap='Blues', alpha=1, extend='max')

# Vectors for Wind difference
q  =  ax[1].quiver(lon, lat, fcesm_men.sel(lev=850)['U_JJA'].data, fcesm_men.sel(lev=850)['V_JJA'].data, 
                    regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                    scale_units='xy', scale=1.95,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                    units='xy', width=0.35,              # width控制粗细
                    transform=proj,
                    color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

add_vector_legend(ax=ax[1], q=q, speed=5)

# --- Coast Line ---
ax[1].coastlines(resolution='110m', lw=1.25)

# ------ CESM - ERA5 Poltting ---------
# --- Tick setting ---
level2 = np.linspace(-15, 15, 11)
set_cartopy_tick(ax=ax[2],extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(10,60,6,dtype=int),nx=1,ny=1,labelsize=25)

im3  =  ax[2].contourf(lon, lat, -1*gaussian_filter((fera5_interp['tp'].data * 1e3 - fcesm_men['PRECT_JJA_BTAL'].data)*1.2, sigma=0.5), levels=level2, cmap='coolwarm_r', alpha=1, extend='both')

# Vectors for Wind difference
q  =  ax[2].quiver(lon, lat, -1*(fera5_interp.sel(level=850)['u'].data - fcesm_men.sel(lev=850)['U_JJA'].data), -1*(fera5_interp.sel(level=850)['v'].data - fcesm_men.sel(lev=850)['V_JJA'].data), 
                    regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                    scale_units='xy', scale=1.95,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                    units='xy', width=0.35,              # width控制粗细
                    transform=proj,
                    color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

add_vector_legend(ax=ax[2], q=q, speed=5)

# --- Coast Line ---
ax[2].coastlines(resolution='110m', lw=1.25)

# ========= add colorbar =================
fig.subplots_adjust(top=0.8) 
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02]) 
cb  =  fig.colorbar(im3, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
#cb.ax.set_xticks(levels)
cb.ax.tick_params(labelsize=17.5)

plt.savefig("/home/sun/paint/ERL/ERL_figs1_model_evaluation_850wind_pr_cb2.pdf")