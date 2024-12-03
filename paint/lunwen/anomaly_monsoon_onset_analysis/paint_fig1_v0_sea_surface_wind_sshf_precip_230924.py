'''
2023-9-24
This script paint fig1 in the article, in order to show the characteristic in the onset early/late years
variables include: surface wind / sensible heat flux / precipitation

reference: paint_composite_ERA5_vorticity_circulation_sensible_flux_230217.py
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
    var[ mask[0] > 0.5] = np.nan

    return var

# ---------------- Paint the Pic - vorticity, wind, sensible_flux ----------------
def paint_composite_wind_sensible_precip(lat_era5, lon_era5, lat_trmm, lon_trmm, u, v, sshf, precip, extent):
    # === Mask the land value ===
    for tttt in range(u[0].shape[0]):
        for p in range(3): #Three situation
            u[p][tttt][ mask_file.lsm.data[0] > 0.4] = np.nan
            v[p][tttt][ mask_file.lsm.data[0] > 0.4] = np.nan
            sshf[p][tttt][ mask_file.lsm.data[0] > 0.4] = np.nan

    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.1 Set the colormap for precipitation
    viridis = cm.get_cmap('Blues', 22)
    newcolors = viridis(np.linspace(0, 1, 22))
    newcmp = ListedColormap(newcolors)
    newcmp.set_under('white')
    newcmp.set_over('#145DA0')

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig1    =  plt.figure(figsize=(34,20))
    spec1   =  fig1.add_gridspec(nrows=3,ncols=4) #First row climate, second early, third late

    # 2.3 Plot the picture
    # --- Firstly, paint the climate ---
    j = 0 ; t = 0
    for col in range(4):
        ax = fig1.add_subplot(spec1[t, col], projection=proj)

        # Tick setting
        set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=25)

        # Equator line
        ax.plot([40,120],[0,0],'k--')

        # Contour for sshf
        im1  =  ax.contour(lon_era5, lat_era5, -1 * sshf[t][j] / 86400 * 24, np.linspace(10,28,7), colors='red',)

        # Shading for precipitation
        im2  =  ax.contourf(lon_trmm, lat_trmm, precip[t][j], np.linspace(5,40,10), cmap=newcmp, alpha=1, extend='max')

        # Coast Line
        ax.coastlines(resolution='110m', lw=1.75)

        # Vector Map
        q  =  ax.quiver(lon_era5, lat_era5, u[t][j], v[t][j], 
            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=1.1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.25,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)

        # Add title of day
        if select_time[j] < 0:
            ax.set_title("D"+str(select_time[j]), loc='right', fontsize=27.5)
        elif select_time[j] > 0:
            ax.set_title("D+"+str(select_time[j]), loc='right',fontsize=27.5)
        else:
            ax.set_title("D"+str(select_time[j]),loc='right',fontsize=27.5)
            
        # Add the Figure number
        # ax.set_title("("+number[j]+")",loc='left',fontsize=27.5)

        # Add legend of the vector
        add_vector_legend(ax=ax,q=q, speed=5)

        j+=1

    # --- Secondly, paint the early situation ---
    j = 0 ; t = 1
    for col in range(4):
        ax = fig1.add_subplot(spec1[t, col], projection=proj)

        # Tick setting
        set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=25)

        # Equator line
        ax.plot([40,120],[0,0],'k--')

        # Contour for sshf
        im1  =  ax.contour(lon_era5, lat_era5, -1 * sshf[t][j] / 86400 * 24, np.linspace(10,28,7), colors='red',)

        # Shading for precipitation
        im2  =  ax.contourf(lon_trmm, lat_trmm, precip[t][j], np.linspace(5,50,10), cmap=newcmp, alpha=1, extend='max')

        # Coast Line
        ax.coastlines(resolution='110m', lw=1.75)

        # Vector Map
        q  =  ax.quiver(lon_era5, lat_era5, u[t][j], v[t][j], 
            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=1.1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.25,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)

        # Add title of day
        if select_time[j] < 0:
            ax.set_title("D"+str(select_time[j]), loc='right', fontsize=27.5)
        elif select_time[j] > 0:
            ax.set_title("D+"+str(select_time[j]), loc='right',fontsize=27.5)
        else:
            ax.set_title("D"+str(select_time[j]),loc='right',fontsize=27.5)
            
        # Add the Figure number
        # ax.set_title("("+number[j]+")",loc='left',fontsize=27.5)

        # Add legend of the vector
        add_vector_legend(ax=ax,q=q, speed=5)

        j+=1

    # --- Thirdly, paint the late situation ---
    j = 0 ; t = 2
    for col in range(4):
        ax = fig1.add_subplot(spec1[t, col], projection=proj)

        # Tick setting
        set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=25)

        # Equator line
        ax.plot([40,120],[0,0],'k--')

        # Contour for sshf
        im1  =  ax.contour(lon_era5, lat_era5, -1 * sshf[t][j] / 86400 * 24, np.linspace(10,28,7), colors='red',)

        # Shading for precipitation
        im2  =  ax.contourf(lon_trmm, lat_trmm, precip[t][j], np.linspace(5,50,10), cmap=newcmp, alpha=1, extend='max')

        # Coast Line
        ax.coastlines(resolution='110m', lw=1.75)

        # Vector Map
        q  =  ax.quiver(lon_era5, lat_era5, u[t][j], v[t][j], 
            regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=1.1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.25,
            transform=proj,
            color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)

        # Add title of day
        if select_time[j] < 0:
            ax.set_title("D"+str(select_time[j]), loc='right', fontsize=27.5)
        elif select_time[j] > 0:
            ax.set_title("D+"+str(select_time[j]), loc='right',fontsize=27.5)
        else:
            ax.set_title("D"+str(select_time[j]),loc='right',fontsize=27.5)
            
        # Add the Figure number
        # ax.set_title("("+number[j]+")",loc='left',fontsize=27.5)

        # Add legend of the vector
        add_vector_legend(ax=ax,q=q, speed=5)

        j+=1

    # ADD colorbar
    fig1.subplots_adjust(top=0.8) 
    cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig1.colorbar(im2, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=25)

    plt.savefig('/home/sun/paint/lunwen/anomoly_analysis/v0_fig1_climate_early_late_circulation_evolve_10mwind_precipitation_sshf.pdf')
# ============================================================================


#===================== First. Read the files =================================
ref_file0  =  xr.open_dataset('/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/single/sst_composite.nc')
mask_file  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

path0      =  '/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/'
path1      =  '/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/single/'

# === U10 wind file name ===
f_uwind = ['u10_composite.nc', 'u10_composite_year_early.nc', 'u10_composite_year_late.nc']
f_vwind = ['v10_composite.nc', 'v10_composite_year_early.nc', 'v10_composite_year_late.nc']

# === Precipitation file name ===
# Here use TRMM
p_precip = '/home/sun/data/composite/'
f_precip = 'trmm_composite_1998_2019_onset_climate_early_late_year.nc'

# === SSHF file name ===
f_sshf = ['sshf_composite.nc', 'sshf_composite_year_early.nc', 'sshf_composite_year_late.nc']

# === Mask file name ===
mask_file  =  xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')

# === Set time to paint ===
select_time = [-6, -4, -2, 0]
number =  ["a","b","c","d"]

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
    # --- Combine the data ---
    # TRMM data
    trmm = xr.open_dataset(p_precip + f_precip).sel(composite_day = select_time)

    # UV wind
    uwind_c = xr.open_dataset(path1 + f_uwind[0]).sel(time = select_time)
    uwind_e = xr.open_dataset(path1 + f_uwind[1]).sel(time = select_time)
    uwind_l = xr.open_dataset(path1 + f_uwind[2]).sel(time = select_time)
    vwind_c = xr.open_dataset(path1 + f_vwind[0]).sel(time = select_time)
    vwind_e = xr.open_dataset(path1 + f_vwind[1]).sel(time = select_time)
    vwind_l = xr.open_dataset(path1 + f_vwind[2]).sel(time = select_time)

    #SSHF
    sshf_c = xr.open_dataset(path1 + f_sshf[0]).sel(time = select_time)
    sshf_e = xr.open_dataset(path1 + f_sshf[1]).sel(time = select_time)
    sshf_l = xr.open_dataset(path1 + f_sshf[2]).sel(time = select_time)

    trmm_combined = [trmm['trmm_climate'].data, trmm['trmm_early'].data, trmm['trmm_late'].data]
    uwind_combined = [uwind_c['u10'].data, uwind_e['u10'].data, uwind_l['u10'].data]
    vwind_combined = [vwind_c['v10'].data, vwind_e['v10'].data, vwind_l['v10'].data]
    sshf_combined  = [sshf_c['sshf'].data, sshf_e['sshf'].data, sshf_l['sshf'].data]

    lonmin,lonmax,latmin,latmax  =  45,115,-10,30
    extent     =  [lonmin,lonmax,latmin,latmax]

    # Deliver parameter to the plot function
    paint_composite_wind_sensible_precip(lat_era5=uwind_c['lat'].data, lon_era5=uwind_c['lon'].data, lat_trmm=trmm['lat'].data, lon_trmm=trmm['lon'].data, u=uwind_combined, v=vwind_combined,
                                         sshf=sshf_combined, precip=trmm_combined, extent=extent)



if __name__ == '__main__':
    main()