'''
2024-7-13
This script is to calculate the varibility of the May precipitation using ERA5
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
from scipy.ndimage import gaussian_filter

# ================ File Information ===================
data_path = '/home/sun/mydown/ERA5/era5_precipitation_daily/'

file_list = os.listdir(data_path) ; file_list.sort()

# subset of 1980 to 2023
file_list_sub  =  file_list[40:-1] 
#print(file_list_sub)

# =====================================================

# =========== Other ============
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

# ================ Calculation ========================
# 1. Extract the May precipitation
start_may = 90 ; end_may = 150 ; start_june = 151 ; end_june = 180
May_precipitation  = np.zeros((len(file_list_sub), 91, 180))
June_precipirarion = np.zeros((len(file_list_sub), 91, 180))

for i in range(len(file_list_sub)):
    #print(file_list_sub[i])
    file0 = xr.open_dataset(data_path + file_list_sub[i])
    May_precipitation[i]  = np.average(file0['tp'].data[start_may:end_may], axis=0)
    June_precipirarion[i] = np.average(file0['tp'].data[start_june:end_june], axis=0)

# 2. Calculate variability
print(np.average(May_precipitation))
may_pt_std = np.std(May_precipitation, axis=0)
jun_pt_std = np.std(June_precipirarion,axis=0)
#print(may_pt_std.shape)

# =============== End of calculation ==================

# =============== Painting ================
proj    =  ccrs.PlateCarree()
fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

# 范围设置
lonmin,lonmax,latmin,latmax  =  65,125,0,30
extent     =  [lonmin,lonmax,latmin,latmax]

# 刻度设置
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60,120,7,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=20)
    
# 绘制赤道线
ax.plot([40,150],[0,0],'--',color='k')

# 绘制降水填色图
sigma = 0.7
im  =  ax.contourf(file0.longitude.data,file0.latitude.data,gaussian_filter(may_pt_std * 1000, sigma=sigma),np.linspace(1,4,13),cmap='Blues',alpha=1,)

# 绘制温度等值线
#it  =  ax.contour(f_ctl.lon.data,f_ctl.lat.data,ts,np.linspace(-10,10,11),alpha=1)

# 绘制海岸线
ax.coastlines(resolution='110m',lw=2)

# 绘制矢量图
#q  =  ax.quiver(f_ctl.lon.data,f_ctl.lat.data, -1*u, -1*v, 
#            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
#            scale_units='xy', scale=.75,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#            units='xy', width=0.35,
#            transform=proj,
#            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
    
# 加序号
#plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))

# 加矢量图图例
#add_vector_legend(ax=ax,q=q, speed=5)

# colorbar
a = fig.colorbar(im,shrink=0.6, pad=0.05,orientation='horizontal')
a.ax.tick_params(labelsize=15)

save_fig(path_out="/home/sun/paint/phd/",file_out="phd_c5_v0_fig1_AprMay_pr_std.pdf")
