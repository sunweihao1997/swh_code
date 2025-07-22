'''
2025-6-30
This script plot the composite anomaly in early and late onset years for pr uv850
'''
import sys
sys.path.append("/home/sun/swh_code/paint")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl
import xarray as xr
from scipy.ndimage import gaussian_filter

import sys
sys.path.append("/home/sun/swh_code/module/")
from module_sun import set_cartopy_tick,add_vector_legend

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

f0 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_composite_anomaly_precip_850wind_onset_day_early_late.nc")

u_climate = f0['u850_climate'].data ; v_climate = f0['v850_climate'].data ; pr_climate = f0['pr_climate'].data
u_early   = f0['u850_early'].data   ; v_early   = f0['v850_early'].data   ; pr_early   = f0['pr_early'].data
u_late    = f0['u850_late'].data    ; v_late    = f0['v850_late'].data    ; pr_late    = f0['pr_late'].data
p_early   = f0['p_early']           ; p_late    = f0['p_early']   

proj    =  ccrs.PlateCarree()
fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

# 范围设置
lonmin,lonmax,latmin,latmax  =  45,135,-10,45
extent     =  [lonmin,lonmax,latmin,latmax]

# 刻度设置
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=19)
    
# 绘制赤道线
ax.plot([40,120],[0,0],'--',color='k')

# 绘制海岸线
ax.coastlines(resolution='110m',lw=1.5)

im  =  ax.contourf(f0.longitude.data, f0.latitude.data, gaussian_filter(100 * ((pr_early - pr_climate) /pr_climate), sigma=1), np.linspace(-100, 100, 11), cmap='coolwarm',alpha=1,extend='both')

im2 = ax.contourf(f0.longitude.data, f0.latitude.data, p_early, [0, 0.1], colors='none', hatches=['..'])
plt.rcParams.update({'hatch.color': 'white'})


# 绘制矢量图
q  =  ax.quiver(f0.longitude.data, f0.latitude.data, u_early - u_climate, v_early - v_climate, 
            regrid_shape=12.5, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.25,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.4,
            transform=proj, 
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)

add_vector_legend(ax=ax, q=q, speed=1)

fig.subplots_adjust(top=0.8) 
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
cb.ax.tick_params(labelsize=20)

plt.savefig("/home/sun/paint/CD/FigureS_supplementary_early_composite_uv850pr.pdf")