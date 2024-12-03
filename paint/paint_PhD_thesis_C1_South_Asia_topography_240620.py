'''
PHD thesis
plot south Asia topography

modified from other script, no worries
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
import xarray as xr
module_path = ["/home/sun/mycode/module/","/data5/2019swh/mycode/module/"]
sys.path.append(module_path[0])
from module_sun import *

ftopo = xr.open_dataset("/home/sun/data/topography/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc")
topo  = ftopo['z'].data[0] / 9.8

#print(np.min(topo))
#print(np.max(topo))
topo[topo<0] = 0 # mask ocean topography



def colormap_from_list_color(list):
    # 本函数读取颜色列表然后制作出来colormap
    return LinearSegmentedColormap.from_list('chaos',list)



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

def paint_picture(lon, lat, u, v, psl):
    # 绘制图像
    proj    =  ccrs.PlateCarree()
    fig,ax   =  plt.subplots(figsize=(25,18.5),subplot_kw=dict(projection=ccrs.PlateCarree()))



    # 范围设置
    lonmin,lonmax,latmin,latmax  =  45,135,-5,45
    extent     =  [lonmin,lonmax,latmin,latmax]

    # 刻度设置
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(30,150,7,dtype=int),yticks=np.linspace(0,40,3,dtype=int),nx=1,ny=1,labelsize=23.5)
    
    # 绘制赤道线
    #ax.plot([40,120],[0,0],'--',color='k')

    im  =  ax.contourf(lon,lat,topo,np.linspace(250,6000,24),cmap='OrRd',alpha=1,extend='max')

    # 绘制海岸线
    ax.coastlines(resolution='50m',lw=3)

    # Add land with green color
    #ax.add_feature(cfeature.LAND, facecolor='gainsboro')

    # Add ocean with blue color
    ax.add_feature(cfeature.OCEAN, facecolor='lightcyan')

#
#
#    # 绘制矢量图
#    q  =  ax.quiver(lon, lat, u, v, 
#                regrid_shape=9, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
#                scale_units='xy', scale=0.75,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#                units='xy', width=0.4,
#                transform=proj,
#                color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
#    
#    # 加序号
#    #plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))
#
#    # 加矢量图图例
#    add_vector_legend(ax=ax,q=q, speed=1)
#
    fig.subplots_adjust(top=0.7) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=20)

    # 保存图片
    save_fig(path_out="/home/sun/paint/monthly_variable/regression/",file_out="emptymap.pdf")

def main():
    f1 = xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_200_uvz_500_w_to_OLR.nc')
    #print(f1)

    paint_picture(lon=ftopo.longitude,lat=ftopo.latitude,u=f1['rc_u'].data*1, v=f1['rc_v'].data*1, psl=f1['rc_w'].data*1000)



if __name__ == "__main__":
    main()
