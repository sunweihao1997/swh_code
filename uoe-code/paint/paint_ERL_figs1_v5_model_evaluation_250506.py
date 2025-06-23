'''
2024-9-19
This script serves for the Figure S1 in ERL
The purpose is to evaluate the simulation of CESM on summer monsoon

variables:
uv 850 ; Precip for JJA

v4: try to use the correct China map
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
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
from matplotlib import cm
from matplotlib.colors import ListedColormap

sys.path.append('/home/sun/swh_code/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# =================== Calculating Part ========================
# 1.1 calculate the JJA mean for ERA5
# !!Note!! Due to ERA5 data not saved on Huaibei Server, this data was generated on ubuntu by script:
# cal_ERL_figs1_model_evaluation_850wind_pr_240919.py
fera5 = xr.open_dataset("/home/sun/data/process/analysis/ERA5/ERA5_monthly_JJA_wind_precip.nc")
fcmap = xr.open_dataset("/home/sun/data/download_data/CMAP/precip.mon.ltm.1991-2020.nc")


# 1.2 calculate the JJA mean for CESM
fcesm_pr = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc")
fcesm_u  = xr.open_dataset("/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTALnEU_JJA_U_ensemble.nc")
fcesm_v  = xr.open_dataset("/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTALnEU_JJA_V_ensemble.nc")
fcesm    = xr.merge([fcesm_pr, fcesm_u, fcesm_v])
fcesm_sel= fcesm.sel(time=slice(1980, 2005))

#print(fcesm_sel)


# 1.2.1 calculate mean for selected period
# calculate the ensemble mean of CESM U and V
u = np.average((fcesm_sel.sel(lev=850)['JJA_U_1'].data + fcesm_sel.sel(lev=850)['JJA_U_2'].data + fcesm_sel.sel(lev=850)['JJA_U_3'].data + fcesm_sel.sel(lev=850)['JJA_U_4'].data + fcesm_sel.sel(lev=850)['JJA_U_5'].data + fcesm_sel.sel(lev=850)['JJA_U_6'].data + fcesm_sel.sel(lev=850)['JJA_U_7'].data + fcesm_sel.sel(lev=850)['JJA_U_8'].data) /8, axis=0)
v = np.average((fcesm_sel.sel(lev=850)['JJA_V_1'].data + fcesm_sel.sel(lev=850)['JJA_V_2'].data + fcesm_sel.sel(lev=850)['JJA_V_3'].data + fcesm_sel.sel(lev=850)['JJA_V_4'].data + fcesm_sel.sel(lev=850)['JJA_V_5'].data + fcesm_sel.sel(lev=850)['JJA_V_6'].data + fcesm_sel.sel(lev=850)['JJA_V_7'].data + fcesm_sel.sel(lev=850)['JJA_V_8'].data) /8, axis=0)
#sys.exit()

# 1.3 Interpolate ERA5 data to the CESM grid
fera5_interp = fera5.interp(lat=fcesm.lat.data, lon=fcesm.lon.data, method='nearest')
fcmap_interp = fcmap.interp(lat=fcesm.lat.data, lon=fcesm.lon.data, method='nearest')
#print(fcesm_men)

# !================== End of calculating =======================!

# ================== Plot Part ================================
# Vector: Wind ; Shading: Precipitation
# Set the figure
proj    =  ccrs.PlateCarree()
fig, ax =  plt.subplots(3, 1, figsize=(15, 25), subplot_kw={'projection': proj})

# coordinate information
lat = fcesm_sel.lat.data ; lon = fcesm_sel.lon.data
levels = np.array([2,4, 6,8,10,12,15, 20, 30])

# --- Set range ---
lonmin,lonmax,latmin,latmax  =  40,125,-20,35
extent     =  [lonmin,lonmax,latmin,latmax]

# ------ ERA5 Poltting ---------
# --- Tick setting ---
set_cartopy_tick(ax=ax[0],extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(-20,30,6,dtype=int),nx=1,ny=1,labelsize=25)

im1  =  ax[0].contourf(lon, lat, gaussian_filter(np.average(fcmap_interp['precip'].data[5:8], axis=0), sigma=1), levels=levels, cmap='Blues', alpha=1, extend='max')

# Vectors for Wind difference
q  =  ax[0].quiver(lon, lat, fera5_interp.sel(level=850)['u'].data, fera5_interp.sel(level=850)['v'].data, 
                    regrid_shape=10, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                    scale_units='xy', scale=1.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                    units='xy', width=0.35,              # width控制粗细
                    transform=proj,
                    color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

add_vector_legend(ax=ax[0], q=q, speed=5)

# --- Coast Line ---
ax[0].coastlines(resolution='110m', lw=1.5)
ax[0].add_feature(cfeature.BORDERS, linewidth=1.)

# ------ CESM Poltting ---------
# --- Tick setting ---
set_cartopy_tick(ax=ax[1],extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(-20,30,6,dtype=int),nx=1,ny=1,labelsize=25)

#print(fcesm_sel['PRECT_JJA_BTAL'].data.shape)
im2  =  ax[1].contourf(lon, lat, gaussian_filter((1*np.average(fcesm_sel['PRECT_JJA_BTAL'].data, axis=0)), sigma=1), levels=levels, cmap='Blues', alpha=1, extend='max')

# Vectors for Wind difference
print(u.shape)
q  =  ax[1].quiver(lon, lat, u, v, 
                    regrid_shape=10, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                    scale_units='xy', scale=1.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                    units='xy', width=0.35,              # width控制粗细
                    transform=proj,
                    color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

add_vector_legend(ax=ax[1], q=q, speed=5)

# --- Coast Line ---
ax[1].coastlines(resolution='110m', lw=1.5)
ax[1].add_feature(cfeature.BORDERS, linewidth=1., edgecolor='black')

# Using China Map
#from cartopy.io.shapereader import Reader
#reader = Reader("/home/sun/data/download_data/shp_file/country/country.shp") #国界
#china_country= cfeature.ShapelyFeature(reader.geometries(), ccrs.PlateCarree(), edgecolor='black', facecolor='none') 
#ax[1].add_feature(china_country, linewidth=1, edgecolor='black') 

# ------ CESM - ERA5 Poltting ---------
# --- Tick setting ---
level2 = np.linspace(-10, 10, 11)
set_cartopy_tick(ax=ax[2],extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(-20,30,6,dtype=int),nx=1,ny=1,labelsize=25)

im3  =  ax[2].contourf(lon, lat, -1*gaussian_filter((np.average(fcmap_interp['precip'].data[5:8], axis=0) - 1*np.average(fcesm_sel['PRECT_JJA_BTAL'].data, axis=0)), sigma=1), levels=level2, cmap='coolwarm_r', alpha=1, extend='both')

# Vectors for Wind difference
q  =  ax[2].quiver(lon, lat, -1*(fera5_interp.sel(level=850)['u'].data - u), -1*(fera5_interp.sel(level=850)['v'].data - v), 
                    regrid_shape=10, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                    scale_units='xy', scale=1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                    units='xy', width=0.35,              # width控制粗细
                    transform=proj,
                    color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

add_vector_legend(ax=ax[2], q=q, speed=5)

# --- Coast Line ---
ax[2].coastlines(resolution='110m', lw=1.5)
ax[2].add_feature(cfeature.BORDERS, linewidth=1., edgecolor='black')

# ========= add colorbar =================
fig.subplots_adjust(top=0.8) 
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.02]) 
cb  =  fig.colorbar(im3, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
#cb.ax.set_xticks(levels)
cb.ax.tick_params(labelsize=25)

plt.savefig("/home/sun/paint/ERL/ERL_figs1_v5_model_evaluation_850wind_pr_cb2_chinamap.pdf")