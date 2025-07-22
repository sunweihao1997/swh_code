'''
2025-7-19
This script is to visualize the wind data from ECMWF.
'''
import pygrib
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl

file0 = pygrib.open('/home/sun/20250718120000-24h-oper-fc.grib2')

variable_names = set()
for grb in file0:
    variable_names.add(grb.shortName)

file0.rewind()  # 确保从头开始读

# 提取 shortName 为 '10u' 的所有消息
u10_msgs = file0.select(shortName='10u')
u10 = u10_msgs[0]  # 取第一条记录

file0.rewind()  # 确保从头开始读

# 提取 shortName 为 '10u' 的所有消息
v10_msgs = file0.select(shortName='10v')
v10 = v10_msgs[0]  # 取第一条记录

#print(u10)              # 显示 GRIB 消息的简要信息
#print(u10.validDate)    # 预报对应的时间
#print(u10.level)        # 层级（应该是10）
data_u, lats, lons = u10.data()
data_v, lats, lons = v10.data()

wind_speed = np.sqrt(data_u**2 + data_v**2)

# ------------------------------- Plot -----------------------------------------
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

def add_vector_legend(ax,q,location=(0.825, 0),length=0.175,wide=0.2,fc='white',ec='k',lw=0.5,order=1,quiver_x=0.915,quiver_y=0.125,speed=10,fontsize=18):
    '''
    句柄 矢量 位置 图例框长宽 表面颜色 边框颜色  参考箭头的位置 参考箭头大小 参考label字体大小
    '''
    rect = mpl.patches.Rectangle((location[0], location[1]), length, wide, transform=ax.transAxes,    # 这个能辟出来一块区域，第一个参数是最左下角点的坐标，后面是矩形的长和宽
                            fc=fc, ec=ec, lw=lw, zorder=order
                            )
    ax.add_patch(rect)

    ax.quiverkey(q, X=quiver_x, Y=quiver_y, U=speed,
                    label=f'{speed} m/s', labelpos='S', labelsep=0.1,fontproperties={'size':fontsize})
    
proj    =  ccrs.PlateCarree()
fig,ax   =  plt.subplots(figsize=(15,12),subplot_kw=dict(projection=ccrs.PlateCarree()))

# 范围设置
lonmin,lonmax,latmin,latmax  =  90,160,0,40
extent     =  [lonmin,lonmax,latmin,latmax]

# 刻度设置
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(90,160,8,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=19)
    
# 绘制赤道线
#ax.plot([40,150],[0,0],'--',color='k')

# 绘制降水填色图
im  =  ax.contourf(lons,lats,wind_speed,np.linspace(2, 26, 13),cmap='tab20b',alpha=1,extend='both')

# 绘制温度等值线
#it  =  ax.contour(f_ctl.lon.data,f_ctl.lat.data,ts,np.array([-1.5, -1, -0.5]),alpha=1, linewidths=2.5, colors='green')
#ax.clabel(it,fontsize=12)
# 绘制海岸线
ax.coastlines(resolution='110m',lw=1.1)

# 绘制矢量图
q  =  ax.quiver(lons,lats,data_u, data_v, 
            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=2,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.35,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
    
# 加序号
#plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))

# 加矢量图图例
add_vector_legend(ax=ax,q=q, speed=5)
a = fig.colorbar(im,shrink=0.6, pad=0.05,orientation='horizontal')
a.ax.tick_params(labelsize=15)

plt.savefig('/home/sun/paint/ECMWF_wind_data_250719.png', dpi=300, bbox_inches='tight')