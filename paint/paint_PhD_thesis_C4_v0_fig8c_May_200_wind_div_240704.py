'''
2024-7-4
This script serves the figure8 in Chapter 4, targetting to show the changes in May moisture transportation
'''
import sys
sys.path.append("/home/sun/mycode/paint")
import paint_lunwen_version3_0_fig1_bob_onset_seris as plv3_1
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import paint_lunwen_version3_0_fig2a_tem_gradient_20220426 as plv3_2a
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl
module_path = ["/home/sun/mycode/module/","/data5/2019swh/mycode/module/"]
sys.path.append(module_path[0])
from module_sun import *
from windspharm.xarray import VectorWind

# =========== File Information ==============
f_ctl = xr.open_dataset("/home/sun/data/model_data/climate/b1850_exp/b1850_control_climate_atmosphere.nc").sel(lev=200)
f_ind = xr.open_dataset("/home/sun/data/model_data/climate/b1850_exp/b1850_indian_climate_atmosphere3.nc").sel(lev=200)
mask_file = xr.open_dataset("/home/sun/data/mask/ERA5_land_sea_mask.nc")
mask_file_interp = mask_file.interp(latitude=f_ctl.lat.data, longitude=f_ctl.lon.data)

u     = np.average(f_ctl['U'].data[120:150], axis=0)     - np.average(f_ind['U'].data[120:150], axis=0)
v     = np.average(f_ctl['V'].data[120:150], axis=0)     - np.average(f_ind['V'].data[120:150], axis=0)

#u[(u**2+v**2)<0.5] = np.nan

disy,disx,location = cal_xydistance(f_ctl.lat.data,f_ctl.lon.data)

uy = np.gradient(u,location,axis=0)
vx = copy.deepcopy(uy)
for i in range(1,191):
    vx[i,:] = np.gradient(v[i,:],disx[i],axis=0)
div = uy+vx

f_qdiv= xr.open_dataset("/home/sun/data/model_data/climate/b1850_exp/b1850_control_Indian_climate_water_transportation_integral_daily_liuxl.nc")

# ===========================================

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

def save_fig(path_out,file_out,dpi=450):
    plv3_1.check_path(path_out)
    plt.savefig(path_out+file_out,dpi=450)

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



# ========= Plot the spatial pattern of difference ============
# 绘制图像
proj    =  ccrs.PlateCarree()
fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

# 范围设置
lonmin,lonmax,latmin,latmax  =  45,130,-10,40
extent     =  [lonmin,lonmax,latmin,latmax]

cmap = plt.cm.coolwarm  # 使用viridis颜色映射
cmap.set_bad(color='gray')

# 刻度设置
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=20)
    
# 绘制赤道线
ax.plot([40,150],[0,0],'--',color='k')

# 绘制divergence填色图
level0 = np.linspace(-10, 10, 11)
im  =  ax.contourf(f_ctl.lon.data,f_ctl.lat.data,div*1e6,level0,cmap='coolwarm',alpha=1,extend='both')
ax.set_facecolor('lightgrey')

# 绘制温度等值线
#it  =  ax.contour(f_ctl.lon.data,f_ctl.lat.data,ts,np.linspace(-10,10,11),alpha=1)

# 绘制海岸线
ax.coastlines(resolution='50m',lw=1.5)

# 绘制矢量图
q  =  ax.quiver(f_ctl.lon.data,f_ctl.lat.data, u, v, 
            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=1.25,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.3,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
    
# 加序号
#plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))

# 加矢量图图例
add_vector_legend(ax=ax,q=q, speed=5)

# colorbar
a = fig.colorbar(im,shrink=0.6, pad=0.05,orientation='horizontal')
a.ax.tick_params(labelsize=15)

# 保存图片
save_fig(path_out="/home/sun/paint/phd/",file_out="phd_c4_fig8c_May_200_uv.pdf")
