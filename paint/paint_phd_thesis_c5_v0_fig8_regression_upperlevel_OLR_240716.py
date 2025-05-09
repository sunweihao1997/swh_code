'''
2023/05/05
This script paint the regression result of u/v to the LSTC index
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

def colormap_from_list_color(list):
    # 本函数读取颜色列表然后制作出来colormap
    return LinearSegmentedColormap.from_list('chaos',list)

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

def paint_picture(lon, lat, u, v, psl, corre):
    # 绘制图像
    proj    =  ccrs.PlateCarree()
    fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))



    # 范围设置
    lonmin,lonmax,latmin,latmax  =  45,135,-10,35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # 刻度设置
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=19)
        
    # 绘制赤道线
    ax.plot([40,120],[0,0],'--',color='k')

    # 绘制海岸线
    ax.coastlines(resolution='110m',lw=1.5)

    im  =  ax.contourf(lon,lat,psl,np.linspace(-2.5,2.5,11),cmap='coolwarm',alpha=1,extend='both')

    im2 = ax.contourf(lon, lat, corre, [0, 0.01], colors='none', hatches=['//'])
    plt.rcParams.update({'hatch.color': 'white'})


    # 绘制矢量图
    q  =  ax.quiver(lon, lat, u, v, 
                regrid_shape=12, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
                scale_units='xy', scale=0.65,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                units='xy', width=0.5,
                transform=proj, 
                color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
    
    # 加序号
    #plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))

    # 加矢量图图例
    add_vector_legend(ax=ax,q=q, speed=5)

    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=20)

    # 保存图片
    save_fig(path_out="/home/sun/paint/phd/",file_out="phd_thesis_C5_fig8_March_200_uv_to_OLR_index.pdf")

def main():
    f1 = xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_200_uvz_500_w_to_OLR_ttest.nc')
    #print(f1)

    paint_picture(lon=f1.lon,lat=f1.lat,u=f1['rc_u'].data*1, v=f1['rc_v'].data*1, psl=f1['rc_w'].data*1e2, corre=f1['prob_d_olr'].data)



if __name__ == "__main__":
    main()
