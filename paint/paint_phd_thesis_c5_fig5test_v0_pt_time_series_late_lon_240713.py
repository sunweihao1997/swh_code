'''
2023-9-24
This script paint fig1 in the article, in order to show the characteristic in the onset early/late years
variables include: surface wind / sensible heat flux / precipitation

reference: paint_composite_ERA5_vorticity_circulation_sensible_flux_230217.py

v1 modified:
Add figure number on the lest
Add type (climate, early, late) on the right
'''
import xarray as xr
import numpy as np
import argparse
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

sys.path.append("/home/sun/mycode/module/")
from module_sun import *

sys.path.append("/home/sun/mycode/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import set_cartopy_tick,save_fig,add_vector_legend

# ==================== Functions =============================================
# ---------------- Mask the value on the land -----------------------------------
def land_sea_mask(var, mask):
    '''This function mask the variable on the land'''
    var[ mask[0] > 0.1] = np.nan

    return var

# ---------------- Paint the Pic - vorticity, wind, sensible_flux ----------------
def paint_composite_wind_sensible_precip(lat_era5, lon_era5, lat_trmm, lon_trmm, u, v, sshf, precip, extent):
    # === Mask the land value ===
    for tttt in range(u[0].shape[0]):
        for p in range(2): #Three situation
            u[p][tttt][ mask_file.lsm.data[0] > 0.1] = np.nan
            v[p][tttt][ mask_file.lsm.data[0] > 0.1] = np.nan
            sshf[p][tttt][ mask_file.lsm.data[0] > 0.1] = np.nan

    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.1 Set the colormap for precipitation
    viridis = cm.get_cmap('Blues', 22)
    newcolors = viridis(np.linspace(0, 1, 22))
    newcmp = ListedColormap(newcolors)
#    newcmp.set_under('white')
#    newcmp.set_over('#145DA0')

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig1    =  plt.figure(figsize=(40,18))
    spec1   =  fig1.add_gridspec(nrows=2,ncols=5) #First row climate, second early, third late

    # 2.3 Plot the picture
    # --- Firstly, paint the early ---
    fig_number1 = ['(a)', '(b)', '(c)', '(d)', '(e)']
    j = 0 ; t = 0
    for col in range(5):
        ax = fig1.add_subplot(spec1[t, col], projection=proj)

        # Tick setting
        set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=20)

        # Equator line
        ax.plot([40,120],[0,0],'k--')

        # Contour for sshf
        #im1  =  ax.contour(lon_era5, lat_era5, -1 * sshf[t][j] / 86400 * 24, np.linspace(10,30,5), colors='red', linewidths=1.5)

        # Shading for precipitation
        #im2  =  ax.contourf(lon_trmm, lat_trmm, precip[t][j], np.linspace(0.5, 4.5, 9), cmap=newcmp, alpha=1,)
        im2  =  ax.contourf(lon_era5, lat_era5, -1 * sshf[t][j] / 86400 * 24, np.linspace(10, 25, 7), cmap='Reds', alpha=1,extend='max')

        # Coast Line
        ax.coastlines(resolution='110m', lw=1.75)

        # Vector Map
        #q  =  ax.quiver(lon_era5, lat_era5, u[t][j], v[t][j], 
        #    regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
        #    scale_units='xy', scale=0.9,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        #    units='xy', width=0.35,
        #    transform=proj,
        #    color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)

        # Add title of day
        ax.set_title(fig_number1[j], loc='left', fontsize=25)
        ax.set_title("D" + str(select_time[j]), loc='right', fontsize=25)
            
        # Add the Figure number
        # ax.set_title("("+number[j]+")",loc='left',fontsize=27.5)

        # Add legend of the vector
        #add_vector_legend(ax=ax,q=q, speed=5)

        j+=1

    # --- Secondly, paint the late situation ---
    fig_number2 = ['(f)', '(g)', '(h)', '(i)', '(j)']
    j = 0 ; t = 1
    for col in range(5):
        ax = fig1.add_subplot(spec1[t, col], projection=proj)

        # Tick setting
        set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=20)

        # Equator line
        ax.plot([40,120],[0,0],'k--')

        # Contour for sshf
        #im1  =  ax.contour(lon_era5, lat_era5, -1 * sshf[t][j] / 86400 * 24, np.linspace(10,28,7), colors='red',)

        # Shading for precipitation
        #im2  =  ax.contourf(lon_trmm, lat_trmm, precip[t][j], np.linspace(0.5, 4.5, 9), cmap=newcmp, alpha=1,)
        im3  =  ax.contourf(lon_era5, lat_era5, -1 * sshf[t][j] / 86400 * 24, np.linspace(12, 25, 7), cmap='Reds', alpha=1,extend='max')

        # Coast Line
        ax.coastlines(resolution='110m', lw=1.75)

        # Vector Map
        #q  =  ax.quiver(lon_era5, lat_era5, u[t][j], v[t][j], 
        #    regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
        #    scale_units='xy', scale=0.9,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        #    units='xy', width=0.35,
        #    transform=proj,
        #    color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)

        # Add title of day
        ax.set_title(fig_number2[j], loc='left', fontsize=25)
        ax.set_title("D" + str(select_time[j]), loc='right', fontsize=25)
            
        # Add the Figure number
        # ax.set_title("("+number[j]+")",loc='left',fontsize=27.5)

        # Add legend of the vector
        #add_vector_legend(ax=ax,q=q, speed=5)

        j+=1


    # ADD colorbar
    fig1.subplots_adjust(top=0.8) 
    cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig1.colorbar(im2, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=25)

    plt.savefig('/home/sun/paint/phd/phd_C5_fig4_v0_climate_early_late_circulation_sshf.pdf')
# ============================================================================

# =================== 


#===================== First. Read the files =================================
ref_file0  =  xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/single/sst_composite.nc')
mask_file  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

path0      =  '/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/'
path1      =  '/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/single/'

# === U10 wind file name ===
f_uwind = ['u10_composite.nc', 'u10_composite_year_early.nc', 'u10_composite_year_late.nc']
f_vwind = ['v10_composite.nc', 'v10_composite_year_early.nc', 'v10_composite_year_late.nc']

# === Precipitation file name ===
# Here use TRMM
p_precip = '/home/sun/data/composite/'
f_precip = 'trmm_composite_1998_2019_onset_climate_early_late_year.nc'

# === SSHF file name ===
f_sshf = ['sshf_composite.nc', 'sshf_composite_year_early.nc', 'sshf_composite_year_late.nc']
f_slhf = ['slhf_composite.nc', 'slhf_composite_year_early.nc', 'slhf_composite_year_late.nc']

# === Mask file name ===
mask_file  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

# === Set time to paint ===
select_time = np.array([-15, -10, -5, -0, 5])
number =  ["a","b","c","d","e"]

## # === Interpolate TRMM data to lower spatial resolution ===
## # --- longitude/latitude Information ---
## # ERA5:(181, 360) 90to-90, 0to359
## # TRMM:(400, 1440) -50to50, -180to180 
## trmm_lat_new = np.linspace(-50, 50, 101)
## trmm_lon_new = np.linspace(-180, 180, 361)
## # Notice: First try the result of non-interpolation

## # === Mask the sshf value over the continent ===



# ==================== Second. Plot the Picture ===================================
def main():
    # === Here I would like to combine all three type together and deliver to the plot function ===
    tfile = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/equavilent_potential_temperature_composite_abnormal.nc")
    tfile_1000 = tfile.sel(lev=1000)
    tfile_925 = tfile.sel(lev=925)
    tfile_850  = tfile.sel(lev=850,)
    tfile_500  = tfile.sel(lev=500,)
    tfile_600  = tfile.sel(lev=600,)

    start0 = 50 ; end0 = 60
    sf_series1 =  tfile_850['theta_e_late'].data[25]
    sf_series2 = tfile_1000['theta_e_late'].data[25]
    sf_series3 =  tfile_500['theta_e_late'].data[25]
    sf_series4 =  tfile_600['theta_e_late'].data[25]
    sf_series5 =  tfile_925['theta_e_late'].data[25]

    diff = sf_series2 - sf_series5
    print(diff.shape)
    #print(sf_series2)
    
    proj    =  ccrs.PlateCarree()
    fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

    # 范围设置
    lonmin,lonmax,latmin,latmax  =  45,150,-10,30
    extent     =  [lonmin,lonmax,latmin,latmax]

    # 刻度设置
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=19)

    # 绘制赤道线
    ax.plot([40,150],[0,0],'--',color='k')

    # 绘制降水填色图
    #im  =  ax.contourf(f_ctl.lon.data,f_ctl.lat.data,pr*86400000,np.linspace(-11, 11, 21),cmap='coolwarm',alpha=1,extend='both')
    #print(diff.shape)
    im  =  ax.contourf(tfile_1000.lon.data,tfile_1000.lat.data,diff,np.linspace(-5, 5, 11),cmap='coolwarm',alpha=1,extend='both')

    # 绘制温度等值线
    #it  =  ax.contour(f_ctl.lon.data,f_ctl.lat.data,ts,np.array([-1.5, -1, -0.5, 0, 0.5, 1, 1.5]),alpha=1, linewidths=2.5, colors='green')
    #ax.clabel(it,fontsize=12)
    # 绘制海岸线
    ax.coastlines(resolution='50m',lw=1.5)



    # 加序号
    #plv3_2a.add_text(ax=ax,string="(b)",fontsize=27.5,location=(0.015,0.91))

    # 加矢量图图例
    #add_vector_legend(ax=ax,q=q, speed=3)

    # colorbar
    a = fig.colorbar(im,shrink=0.6, pad=0.05,orientation='horizontal')
    a.ax.tick_params(labelsize=15)

    #ax.set_title("No_Inch", loc='left', fontsize=22)
    #ax.set_title("SST", loc='right', fontsize=22)

    # 保存图片  
    save_fig(path_out="/home/sun/paint/phd/",file_out="phd_c5_test_diff_pt_1000_925.pdf")

    
    #    lf_series = np.average(np.average(slhf_e['slhf']/-86400*24, axis=1), axis=1)
    #    p_series = np.average(np.average(olr_f['olr_early']*86400*24, axis=1), axis=1)
    #    print(lf_series)
    #    fig, ax1 = plt.subplots(figsize=(20, 10))
    #    ax1.plot(sf_series1[60:90], color='k')
    #    ax1.plot(sf_series2[60:90], color='r')
    #    ax1.plot(sf_series3[60:90], color='b')
    ##    ax1.set_ylim((90, 180))
    ##    #ax1.tick_params(axis='y', labelcolor=color)
    ##
    ##    ax2 = ax1.twinx()
    ##    ax2.plot(sf_series, color='r')
    ##    ax2.set_ylim((10, 25))
    ##
    ##    ax3 = ax1.twinx()
    ##    ax3.spines['right'].set_position(('outward', 60))
    ##    ax3.plot(u_series, color='k')
    ##    ax3.set_ylim((0, 5))
    #
    #    plt.savefig('u1_u2_late.png')
    #
    #
if __name__ == '__main__':
    main()