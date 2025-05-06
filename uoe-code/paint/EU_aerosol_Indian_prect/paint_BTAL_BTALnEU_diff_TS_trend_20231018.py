'''
2023-10-18
This script paint the difference in TS trend between control experiment and noEU experiment
Variable is TS
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


# === Calculate Difference between two experiments ===
file0 = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_surface_temperature_jja_mean_diff_1900_1960_231017.nc')
file1 = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTALnEU_surface_temperature_jja_mean_diff_1900_1960_231018.nc')

diff_ts  =  file0['TS_JJA_DIFF'].data - file1['TS_JJA_DIFF'].data

# === Paint the Picture ===============================
# ======== Set Extent ==========
#lonmin,lonmax,latmin,latmax  =  0,180,-20,75
lonmin,lonmax,latmin,latmax  =  40,110,-10,40
extent     =  [lonmin,lonmax,latmin,latmax]

# ======== Set Figure ==========
proj       =  ccrs.PlateCarree()

ax         =  plt.subplot(projection=proj)

# Tick setting
cyclic_data, cyclic_lon = add_cyclic_point(diff_ts, coord=file0['lon'].data)
#set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(20,180,9,dtype=int),yticks=np.linspace(-20,60,5,dtype=int),nx=1,ny=1,labelsize=12)
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=12)

im2  =  ax.contourf(cyclic_lon, file0['lat'].data, cyclic_data, np.linspace(-0.6,0.6,13), cmap='coolwarm', alpha=1, extend='both')

ax.coastlines(resolution='10m', lw=1.)

bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='10m', facecolor='none', alpha=0.7)
ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

ax.set_title('1901-1921 to 1941-1961', fontsize=15)

# Add colorbar
plt.colorbar(im2, orientation='horizontal')

plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/TS/BTAL_BTALnEU_diff_JJA_Indian_TS_small_scale.pdf')