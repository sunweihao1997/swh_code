'''
2025-5-13
This script is to pplot the land-sea mask of South Asia
'''
import xarray as xr
import numpy as np
import os
import sys
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
from matplotlib.path import Path
import matplotlib.patches as patches

sys.path.append('/home/sun/swh_code/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

boundary_line = np.array([30, -30, 50, 150]) 

landsea_file = xr.open_dataset("/home/sun/wd_14/download_reanalysis/ERA5/invariant/land_sea_mask.nc").sel(longitude=slice(boundary_line[2], boundary_line[3]), latitude=slice(boundary_line[0], boundary_line[1]))
ls_mask      = landsea_file['lsm'].data[0]

# =========== Other ============
def save_fig(path_out,file_out,dpi=450):
    plt.savefig(path_out+file_out,dpi=450)

def paint_picture(lon, lat, lsm,):
    # 绘制图像
    proj    =  ccrs.PlateCarree()
    fig,ax   =  plt.subplots(figsize=(15,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

    # 范围设置
    lonmin,lonmax,latmin,latmax  =  50,150,-20,30
    extent     =  [lonmin,lonmax,latmin,latmax]

    # 刻度设置
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,150,6,dtype=int),yticks=np.linspace(-30,30,3,dtype=int),nx=1,ny=1,labelsize=20)
    

    im  =  ax.contourf(lon,lat,lsm,np.array([0, 1]),cmap='coolwarm',alpha=1,extend='both')

    # 保存图片
    save_fig(path_out="/home/sun/paint/paint_beijing/",file_out="landsea_mask.png",dpi=450)

paint_picture(lon=landsea_file['longitude'].data,lat=landsea_file['latitude'].data,lsm=ls_mask)