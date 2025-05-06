'''
2024-1-18
This script is to plot the period difference of the SST between 1900-1920 and 1940-1960
The aim is to compare the long-term changes between Model and ERSST
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

sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import cal_xydistance

# ======================= Read file =========================================

src_path = '/exports/csce/datastore/geos/users/s2618078/data/download/ERSST/'
src_file = 'sst.mnmean.nc'

f0       = xr.open_dataset(src_path + src_file)

#print(f0) # Start from 1854 year, total 2040 length (170 years)
lat        = f0['lat'].data
lon        = f0['lon'].data
time       = f0['time'].data
num_years  = int(len(time) / 12)
# ==========================================================================

# ======================= Calculation : 1. calculate JJAS mean ===========================
# Following is the part of calculation which output the ncfile so it has been marked

#   # 1.1 Claim the array
#   JJAS_SST = np.zeros((num_years, len(lat), len(lon)))
#   
#   f0_JJAS  = f0.sel(time=f0.time.dt.month.isin([6, 7, 8, 9]))
#   
#   # 1.2 Calculation
#   #print(f0_JJAS.time.shape)
#   for tt in range(num_years):
#       JJAS_SST[tt]  =  np.nanmean(f0_JJAS['sst'].data[tt*4:tt*4+4], axis=0)
#   
#   # 1.3 Write to the ncfile
#   ncfile  =  xr.Dataset(
#   {
#       "SST_JJAS":     (["time", "lat", "lon"], JJAS_SST),
#   },
#   coords={
#       "time": (["time"], np.linspace(1854, 1854 + 169, 170)),
#       "lat":  (["lat"],  lat),
#       "lon":  (["lon"],  lon),
#   },
#   )
#   
#   ncfile['SST_JJAS'].attrs = f0['sst'].attrs
#   
#   ncfile.attrs['description']  =  'Created on 2024-1-18. This file is the JJAS-mean sea surface temperature calculated from ERSST.'
#   ncfile.attrs['script']       =  'paint_ERL_figS4_ERSST_period_difference_240118.py'
#   
#   ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/ERSST_JJAS_SST_1854_170years.nc", format='NETCDF4')

# =========================================================================================

# ======================== Calculation : 2. calculate the period difference ========================

f1  =  xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/ERSST_JJAS_SST_1854_170years.nc")

period1 = slice(1900, 1920)
period2 = slice(1940, 1960)

f1_period1 = f1.sel(time=period1)
f1_period2 = f1.sel(time=period2)

# 2.1 calculate period difference

diff_sst = np.average(f1_period2['SST_JJAS'].data, axis=0) - np.average(f1_period1['SST_JJAS'].data, axis=0)

# ===================================================================================================

# ========================= Paint : Period difference of SST ========================================

def paint_jjas_diff_BTAL(sst):
    '''This function paint the Diff aerosol JJA'''
    proj       =  ccrs.PlateCarree()
    ax         =  plt.subplot(projection=proj)

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  30,110,-10,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=10.5)

    # contourf for the geopotential height
    im1  =  ax.contourf(lon, lat, sst, levels=np.linspace(-0.5, 0.5, 21), cmap='coolwarm', alpha=1, extend='both')

    ax.coastlines(resolution='110m', lw=1.1)
    ax.add_feature(cartopy.feature.LAND, zorder=100, facecolor='white')

    ax.set_title('ERSST', loc='right', fontsize=12.5)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/ERSST_diff_JJAS_SST.pdf')

paint_jjas_diff_BTAL(diff_sst)