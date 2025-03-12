'''
2024-1-25
This script is to plot the difference between early composite and late composite at lower level
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

# ================ Settings ======================
#    month_selection = np.array([3,4,]) # This setting represents which months will be used to plot pre monsoon background
#    
#    
#    # ================ File Location =================
#    files_location = "/home/sun/mydown/ERA5/monthly_single/"
#    vars_list      = ["u10", "v10", "skt", "tp", "msl"]
#    onset_file     = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")
#    
#    # ================ 1. Read files =================
#    # 1.1 get file list
#    file_list      = os.listdir(files_location)
#    file_list_nc   = [x for x in file_list if x[-2:] == "nc"]
#    #print(file_list_nc)
#    
#    # 1.2  onset-date data
#    onset_day_all   = onset_file["onset_day"].data
#    onset_day_early = onset_file["onset_day_early"]
#    onset_day_late  = onset_file["onset_day_late"]
#    
#    #sys.exit()
#    
#    # ============== 2. Calculation ============
#    # 2.1 Claim the array
#    # 2.1.1 get the anomaly years number
#    #onset_std   = np.std(onset_day_all) * 0.8
#    
#    climate_sst = np.zeros((42, 721, 1440)) ; early_sst = np.zeros((6, 721, 1440)) ; late_sst = np.zeros((9, 721, 1440))
#    climate_u10 = np.zeros((42, 721, 1440)) ; early_u10 = np.zeros((6, 721, 1440)) ; late_u10 = np.zeros((9, 721, 1440))
#    climate_v10 = np.zeros((42, 721, 1440)) ; early_v10 = np.zeros((6, 721, 1440)) ; late_v10 = np.zeros((9, 721, 1440))
#    climate_tp  = np.zeros((42, 721, 1440)) ; early_tp  = np.zeros((6, 721, 1440)) ; late_tp  = np.zeros((9, 721, 1440))
#    climate_slp = np.zeros((42, 721, 1440)) ; early_slp = np.zeros((6, 721, 1440)) ; late_slp = np.zeros((9, 721, 1440))
#    
#    # 2.2 Allocate the value respectively
#    count_climate = 0 ; count_early = 0 ; count_late = 0
#    for yyyy in range(1980, 2022):
#        # 2.2.1 climate value
#        # read the file
#        f_single = xr.open_dataset(files_location + str(int(yyyy)) + "_single_month.nc")
#    
#        f_single_spring = f_single.sel(time=f_single.time.dt.month.isin(month_selection))
#    
#        #print(f_single_spring[vars_list[0]].data.shape)
#        climate_sst[count_climate] = np.average(f_single_spring['skt'].data, axis=0)
#        climate_u10[count_climate] = np.average(f_single_spring['u10'].data, axis=0)
#        climate_v10[count_climate] = np.average(f_single_spring['v10'].data, axis=0)
#        climate_tp[count_climate]  = np.average(f_single_spring['tp'].data, axis=0)
#        climate_slp[count_climate]  = np.average(f_single_spring['msl'].data, axis=0)
#        count_climate += 1
#    
#        # 2.2.2 early onset value
#        if yyyy in onset_day_early.year_early.data:
#            print(f"Detect early onset years which is {yyyy}")
#            early_sst[count_early] = np.average(f_single_spring['skt'].data, axis=0)
#            early_u10[count_early] = np.average(f_single_spring['u10'].data, axis=0)
#            early_v10[count_early] = np.average(f_single_spring['v10'].data, axis=0)
#            early_tp[count_early]  = np.average(f_single_spring['tp'].data, axis=0)
#            early_slp[count_early]  = np.average(f_single_spring['msl'].data, axis=0)
#            count_early += 1
#    
#        if yyyy in onset_day_late.year_late.data:
#            print(f"Detect late onset years which is {yyyy}")
#            late_sst[count_late] = np.average(f_single_spring['skt'].data, axis=0)
#            late_u10[count_late] = np.average(f_single_spring['u10'].data, axis=0)
#            late_v10[count_late] = np.average(f_single_spring['v10'].data, axis=0)
#            late_tp[count_late]  = np.average(f_single_spring['tp'].data, axis=0)
#            late_slp[count_late]  = np.average(f_single_spring['msl'].data, axis=0)
#            count_late += 1
#    
#    # calculate the student-t test of the early and late onset composite
#    p_array = np.ones((721, 1440)) # Here I plan to useu10 or v10 to represent the significance
#    
#    for i in range(721):
#        for j in range(1440):
#            t_stat1, p_valu1 = ttest_ind(early_u10[:, i, j], late_u10[:, i, j])
#            t_stat2, p_valu2 = ttest_ind(early_v10[:, i, j], late_v10[:, i, j])
#    
#            if p_valu1 < 0.15 or p_valu2 < 0.15:
#                p_array[i, j] = min(p_valu1, p_valu2)
#            else:
#                continue
#    
#    # 2.3 calculate the difference
#    diff_sst = np.average(early_sst, axis=0) - np.average(late_sst, axis=0) ; diff_u10 = np.average(early_u10, axis=0) - np.average(late_u10, axis=0) ; diff_v10 = np.average(early_v10, axis=0) - np.average(late_v10, axis=0) ; diff_tp = np.average(early_tp, axis=0) - np.average(late_tp, axis=0) ; diff_slp = np.average(early_slp, axis=0) - np.average(late_slp, axis=0)
#    
#    # 2.4 investigate the unit
#    print(np.nanmean(early_sst))
#    print(np.nanmean(climate_tp))
#    
#    # 2.5 Post-processing
#    # 2.5.1 Mask file
#    mask_file = xr.open_dataset("/home/sun/data/mask/ERA5_land_sea_mask.nc")
#    for i in range(721):
#        for j in range(1440):
#            if mask_file['lsm'].data[0, i, j] > 0.1:
#                diff_sst[i, j] = np.nan ; diff_tp[i, j] = np.nan
#                diff_u10[i, j] = np.nan ; diff_v10[i, j] = np.nan
#    
#    
#    # 2.6 Write to ncfile
#    ncfile  =  xr.Dataset(
#        {
#            "diff_sst":     (["latitude", "longitude"], diff_sst),     
#            "diff_tp":      (["latitude", "longitude"], diff_tp),     
#            "diff_u10":     (["latitude", "longitude"], diff_u10),     
#            "diff_v10":     (["latitude", "longitude"], diff_v10),
#            "diff_slp":     (["latitude", "longitude"], diff_slp),
#            "diff_pvalue":  (["latitude", "longitude"], p_array)     
#        },
#        coords={
#            "latitude":   (["latitude"],  f_single.latitude.data),
#            "longitude":  (["longitude"], f_single.longitude.data),
#        },
#        )
#    
#    ncfile.attrs['description'] = 'Created on 2025-1-27 by /home/sun/swh_code/paint/paint_Article_v2_fig5a_composite_difference_spatial_variables_240125.py. This file save the difference between the early and late composite. while the p value is the significance value of the u10/v10. This is the March value.'
#    
#    
#    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/" + "composite_difference_u10_v10_tp_sst_slp_MA.nc")

#sys.exit()
# ================== 3. Painting ==========================
f0      =  xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/composite_difference_u10_v10_tp_sst_slp_MA.nc")
u       =  f0["diff_u10"].data
v       =  f0["diff_v10"].data

# 绘制图像
proj    =  ccrs.PlateCarree()
fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

lon     =  f0.longitude.data ; lat    =  f0

u_effective = u.copy() ; u_ineffective = u.copy()
u_effective[f0["diff_pvalue"].data > 0.025] = np.nan
u_ineffective[f0["diff_pvalue"].data < 0.025] = np.nan

# 范围设置
lonmin,lonmax,latmin,latmax  =  50,120,-10,27.5
extent     =  [lonmin,lonmax,latmin,latmax]

# 刻度设置
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=20)
    
# 绘制赤道线
ax.plot([40,120],[0,0],'--',color='grey')

# 绘制海岸线
ax.coastlines(resolution='110m',lw=1)

im1  =  ax.contourf(f0.longitude.data, f0.latitude.data, f0['diff_slp'].data, np.linspace(-200, 200, 11), cmap='coolwarm', alpha=1, extend='both')
#im2  =  ax.contour(f0.longitude.data, f0.latitude.data,  f0['diff_tp'].data*1000/31*24,  levels=np.linspace(-5, 5, 11), colors='black', linewidths=1., alpha=1, zorder=1)

#im3  = ax.contourf(f0.longitude.data, f0.latitude.data, f0['diff_pvalue'].data, [0, 0.1], colors='none', hatches=['.'])


# 绘制矢量图
q  =  ax.quiver(f0.longitude.data, f0.latitude.data, u_effective, v, 
            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.4,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.35,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)

q2  =  ax.quiver(f0.longitude.data, f0.latitude.data, u_ineffective, v, 
            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.35,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.35,
            transform=proj,
            color='silver',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5)
    
# 加序号
#plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))

# 加矢量图图例
add_vector_legend(ax=ax,q=q, speed=1)

fig.subplots_adjust(top=0.8) 
cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
cb  =  fig.colorbar(im1, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
cb.ax.tick_params(labelsize=20)

# 保存图片
save_fig(path_out="/home/sun/paint/monsoon_onset_composite_ERA5/", file_out="Article_Anomaly_ISO_v2_fig5a_surface_difference_March_and_April_v2.pdf")
