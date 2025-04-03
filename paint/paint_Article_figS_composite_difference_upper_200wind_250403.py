'''
2025-4-3
This script is to plot the difference between early composite and late composite at upper level for variables UV200 divergence Omega
'''
import xarray as xr
import numpy as np
import argparse
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import math
from scipy.stats import ttest_ind

sys.path.append("/home/sun/mycode/module/")
from module_sun import *

sys.path.append("/home/sun/mycode/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import set_cartopy_tick,save_fig,add_vector_legend

# ================== 3. Painting ==========================
f0      =  xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/Composite_difference_early_late_onset_years_u200_v200_w500.nc")
f_infer =  xr.open_dataset("/home/sun/data/ERA5_SST/mon/ERA5_pentad_month_SST_1959-2021.nc")

#f0      =  f0.interp(lat=f_infer.lat.data, lon=f_infer.lon.data)
u       =  f0["diff_u"].data
v       =  f0["diff_v"].data
w       =  f0["diff_w"].data

#print(np.average(w))
#sys.exit()

# 绘制图像
proj    =  ccrs.PlateCarree()
fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

lon     =  f0.lon.data ; lat    =  f0.lat.data

#u_effective = u.copy() ; u_ineffective = u.copy()
#u_effective[f0["p_u"].data > 0.05] = np.nan
#u_ineffective[f0["p_u"].data < 0.05] = np.nan

# 范围设置
lonmin,lonmax,latmin,latmax  =  45,150,-10,35
extent     =  [lonmin,lonmax,latmin,latmax]

# 刻度设置
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=19)
    
# 绘制赤道线
ax.plot([40,150],[0,0],'--',color='grey')

# 绘制海岸线
ax.coastlines(resolution='110m',lw=1)

im1  =  ax.contourf(f0.lon.data, f0.lat.data, f0['diff_w'].data * 1e2, np.linspace(-10, 10, 11), cmap='coolwarm', alpha=1, extend='both')
#im2  =  ax.contour(f0.longitude.data, f0.latitude.data,  f0['diff_tp'].data*1000/31*24,  levels=np.linspace(-5, 5, 11), colors='black', linewidths=1., alpha=1, zorder=1)

#im3  = ax.contourf(f0.longitude.data, f0.latitude.data, f0['diff_pvalue'].data, [0, 0.1], colors='none', hatches=['.'])


# 绘制矢量图
q  =  ax.quiver(f0.lon.data, f0.lat.data, u, v, 
            regrid_shape=10, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.75,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.5,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)

#q2  =  ax.quiver(f0.lon.data, f0.lat.data, u_ineffective, v, 
#            regrid_shape=10, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
#            scale_units='xy', scale=0.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#            units='xy', width=0.35,
#            transform=proj,
#            color='silver',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
    
# 加序号
#plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))

# 加矢量图图例
add_vector_legend(ax=ax,q=q, speed=5)

fig.subplots_adjust(top=0.8) 
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
cb  =  fig.colorbar(im1, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
cb.ax.tick_params(labelsize=20)

# 保存图片
save_fig(path_out="/home/sun/paint/lunwen/anomoly_analysis/", file_out="Article_Anomaly_composite_diff_200wind.pdf")
