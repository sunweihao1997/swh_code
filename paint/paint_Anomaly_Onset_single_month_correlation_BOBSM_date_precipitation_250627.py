'''
2025-6-27
This script is to plot the correlationship between BOBSM onset date and monthly precipitation
'''
import sys
sys.path.append("/home/sun/swh_code/paint")
#import paint_lunwen_version3_0_fig1_bob_onset_seris as plv3_1
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
#import paint_lunwen_version3_0_fig2a_tem_gradient_20220426 as plv3_2a
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl
import xarray as xr
#module_path = ["/home/sun/swh_code/module/","/data5/2019swh/swh_code/module/"]
#sys.path.append(module_path[0])
#from module_sun import *

corr_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_correlation_BOBSMonset_monthly_precipitation_lowresolution.nc")

def set_cartopy_tick(ax, extent, xticks, yticks, nx=0, ny=0,
    xformatter=None, yformatter=None,labelsize=20):
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    # 本函数设置地图上的刻度 + 地图的范围
    proj = ccrs.PlateCarree()
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    # 设置次刻度.
    xlocator = mticker.AutoMinorLocator(nx + 1)
    ylocator = mticker.AutoMinorLocator(ny + 1)
    ax.xaxis.set_minor_locator(xlocator)
    ax.yaxis.set_minor_locator(ylocator)

    # 设置Formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    # 设置axi label_size，这里默认为两个轴
    ax.tick_params(axis='both',labelsize=labelsize)

    # 在最后调用set_extent,防止刻度拓宽显示范围.
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=proj)

# ================== Starting plot ======================
proj    =  ccrs.PlateCarree()
fig, axs = plt.subplots(2, 2, figsize=(13, 10),
                        subplot_kw=dict(projection=ccrs.PlateCarree()))

# 范围设置
lonmin,lonmax,latmin,latmax  =  45,135,-10,45
extent     =  [lonmin,lonmax,latmin,latmax]

# ------------- 1. The first Pannel --------------
# 刻度设置
set_cartopy_tick(ax=axs[0, 0],extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=19)

# 绘制赤道线
axs[0, 0].plot([40,120],[0,0],'--',color='k')

# 绘制海岸线
axs[0, 0].coastlines(resolution='110m',lw=1.5)

im  =  axs[0, 0].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['correlation_jun'].data, np.linspace(-1, 1, 11),cmap='coolwarm',alpha=1,extend='both')

im2 =  axs[0, 0].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['p_jun'].data, [0, 0.1], colors='none', hatches=['..'])

# ------------- 2. The Second Pannel --------------
# 刻度设置
set_cartopy_tick(ax=axs[0, 1],extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=19)

# 绘制赤道线
axs[0, 1].plot([40,120],[0,0],'--',color='k')

# 绘制海岸线
axs[0, 1].coastlines(resolution='110m',lw=1.5)

im  =  axs[0, 1].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['correlation_jul'].data, np.linspace(-1, 1, 11),cmap='coolwarm',alpha=1,extend='both')

im2 =  axs[0, 1].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['p_jul'].data, [0, 0.1], colors='none', hatches=['..'])

# ------------- 3. The Third Pannel --------------
# 刻度设置
set_cartopy_tick(ax=axs[1, 0],extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=19)

# 绘制赤道线
axs[1, 0].plot([40,120],[0,0],'--',color='k')

# 绘制海岸线
axs[1, 0].coastlines(resolution='110m',lw=1.5)

im  =  axs[1, 0].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['correlation_aug'].data, np.linspace(-1, 1, 11),cmap='coolwarm',alpha=1,extend='both')

im2 =  axs[1, 0].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['p_aug'].data, [0, 0.1], colors='none', hatches=['..'])

# ------------- 4. The Forth Pannel --------------
# 刻度设置
set_cartopy_tick(ax=axs[1, 1],extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=19)

# 绘制赤道线
axs[1, 1].plot([40,120],[0,0],'--',color='k')

# 绘制海岸线
axs[1, 1].coastlines(resolution='110m',lw=1.5)

im  =  axs[1, 1].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['correlation_sep'].data, np.linspace(-1, 1, 11),cmap='coolwarm',alpha=1,extend='both')

im2 =  axs[1, 1].contourf(corr_file['longitude'].data, corr_file['latitude'].data, corr_file['p_sep'].data, [0, 0.1], colors='none', hatches=['..'])

plt.savefig("/home/sun/paint/CD/FigureS_supplementary_correlation_BOBSM_precipitation.pdf")