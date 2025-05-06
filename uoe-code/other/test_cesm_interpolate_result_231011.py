import xarray as xr
import numpy as np
import argparse
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

f0 = xr.open_dataset('/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/U_BTALnEU_3.cam.h0.1960-01--1969-12.nc')
f1 = xr.open_dataset('/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/V_BTALnEU_3.cam.h0.1960-01--1969-12.nc')

proj    =  ccrs.PlateCarree()
fig1    =  plt.figure(figsize=(34,20))
spec1   =  fig1.add_gridspec(nrows=3,ncols=3)

j = 0 ; t = 0
for row in range(3):
    for col in range(3):
        ax = fig1.add_subplot(spec1[row, col], projection=proj)

        # Coast Line
        ax.coastlines(resolution='110m', lw=1.75)

        # Vector Map
        q  =  ax.quiver(f0['lon'].data, f0['lat'].data, f0['U'].data[j, 20], f1['V'].data[j, 20], 
            regrid_shape=25, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=1.1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.25,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)


        j+=1

plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/test1.png')