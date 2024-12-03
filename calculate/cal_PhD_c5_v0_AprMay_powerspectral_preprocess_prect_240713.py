'''
2024-7-13
This script is to preprocess the area-averaged data of precipitation for power spectral analysis
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



# ================ Calculation ========================
# 1. Extract the May precipitation
start_may = 90 ; end_may = 150
aprmay_precipitation  = np.array([])


for i in range(len(file_list_sub)):
    #print(file_list_sub[i])
    file0    = xr.open_dataset(data_path + file_list_sub[i]).sel(latitude=slice(20, 10), longitude=slice(90, 100))
    area_avg = np.average(np.average(file0['tp'].data[90:150], axis=1), axis=1) * 1e3
    #print(area_avg.shape)

    aprmay_precipitation = np.append(aprmay_precipitation, area_avg)

# Write to ncfile
# ----------- save to the ncfile ------------------
ncfile  =  xr.Dataset(
{
    "tp_series": (["time"], aprmay_precipitation),
},
coords={
    "time": (["time"], np.linspace(0, aprmay_precipitation.shape[0], aprmay_precipitation.shape[0])),
},
)
ncfile.attrs['description']  =  'precipitation area average over BOB in April-May'
ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_prect_Apr_May_area_average.nc")


# =============== End of calculation ==================

## =============== Painting ================
#proj    =  ccrs.PlateCarree()
#fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))
#
## 范围设置
#lonmin,lonmax,latmin,latmax  =  55,125,-10,30
#extent     =  [lonmin,lonmax,latmin,latmax]
#
## 刻度设置
#set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60,120,7,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=20)
#    
## 绘制赤道线
#ax.plot([40,150],[0,0],'--',color='k')
#
## 绘制降水填色图
#sigma = 0.7
#im  =  ax.contourf(file0.longitude.data,file0.latitude.data,gaussian_filter(may_pt_std * 1000, sigma=sigma),np.linspace(1,4,13),cmap='Blues',alpha=1,)
#
## 绘制温度等值线
##it  =  ax.contour(f_ctl.lon.data,f_ctl.lat.data,ts,np.linspace(-10,10,11),alpha=1)
#
## 绘制海岸线
#ax.coastlines(resolution='110m',lw=2)
#
## 绘制矢量图
##q  =  ax.quiver(f_ctl.lon.data,f_ctl.lat.data, -1*u, -1*v, 
##            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
##            scale_units='xy', scale=.75,        # scale是参考矢量，所以取得越大画出来的箭头就越短
##            units='xy', width=0.35,
##            transform=proj,
##            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
#    
## 加序号
##plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))
#
## 加矢量图图例
##add_vector_legend(ax=ax,q=q, speed=5)
#
## colorbar
#a = fig.colorbar(im,shrink=0.6, pad=0.05,orientation='horizontal')
#a.ax.tick_params(labelsize=15)
#
#save_fig(path_out="/home/sun/paint/phd/",file_out="phd_c5_v0_fig1_AprMay_pr_std.pdf")
