'''
2023-11-7
This script use windspharm module to calculate divergent wind, the result calculated by ncl is wrong
'''
from windspharm.xarray import VectorWind
import numpy as np
import xarray as xr
from matplotlib import projections
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cartopy
import cmasher as cmr

sys.path.append("/home/sun/mycode/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import set_cartopy_tick,save_fig,add_vector_legend

lonmin,lonmax,latmin,latmax  =  45,115,-10,30
extent     =  [lonmin,lonmax,latmin,latmax]


file_path = '/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/'
u_file    = 'u_component_of_wind_composite_abnormal.nc'
v_file    = 'v_component_of_wind_composite_abnormal.nc'

u         =  xr.open_dataset(file_path + u_file).sel(lev=150)
v         =  xr.open_dataset(file_path + v_file).sel(lev=150)

w_early   =  VectorWind(u['u_early'], v['v_early'])
w_late    =  VectorWind(u['u_late'],  v['v_late'])

#print(w.vorticity())
du_e, dv_e    =  w_early.irrotationalcomponent()
du_l, dv_l    =  w_late.irrotationalcomponent()

div_e         =  w_early.divergence()
div_l         =  w_late.divergence()

#print(div_l)
#  Calculate difference on the onset pentad
diff_div      =  np.average(div_l.sel(time=slice(0, 5)).data, axis=0) - np.average(div_e.sel(time=slice(0, 5)).data, axis=0)
diff_u        =  np.average(du_l.sel(time=slice(0, 5)).data, axis=0)  - np.average(du_e.sel(time=slice(0, 5)).data, axis=0)
diff_v        =  np.average(dv_l.sel(time=slice(0, 5)).data, axis=0)  - np.average(dv_e.sel(time=slice(0, 5)).data, axis=0)

# ========== Settings ===============
    # cmap use cmasher
cmap = cmr.prinsenvlag_r
# levels
level = np.linspace(-1.5, 1.5, 11)

# ========== Painting ===============
proj         =  ccrs.PlateCarree()
fig1, ax    =  plt.subplots(figsize=(15,12),  subplot_kw={'projection': ccrs.PlateCarree()})



# Tick setting
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=22)

# Equator line
ax.plot([40,120],[0,0],'k--')

# Shading for precipitation
im  =  ax.contourf(u.lon.data, v.lat.data, diff_div * 1e5, levels=level, cmap=cmap, alpha=1, extend='both')

# Coast Line
ax.coastlines(resolution='110m', lw=1.75)

q  =  ax.quiver(u['lon'].data, u['lat'].data, diff_u, diff_v, 
        regrid_shape=12, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=1.1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.45,
        transform=proj,
        color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)
add_vector_legend(ax=ax,q=q, speed=5)

# 3. Color Bar
fig1.subplots_adjust(top=0.8) 
cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
cb  =  fig1.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
cb.ax.tick_params(labelsize=25)

plt.savefig('/home/sun/paint/lunwen/anomoly_analysis/v0_figs1_upper_level_divergence_diff.pdf', dpi=500)
#from matplotlib import cm
#from matplotlib.colors import ListedColormap
#import cmasher as cmr
## ========== Settings ===============
## cmap use cmasher
#cmap = cmr.prinsenvlag_r
## levels
#level = np.linspace(-3., 3., 13)
## ========== Painting ===============
#proj    =  ccrs.PlateCarree()
#fig1    =  plt.figure(figsize=(34,20))
#spec1   =  fig1.add_gridspec(nrows=1,ncols=4) #First row early, second late
## 1. Plot the early year
#fig_number = [24, 26, 28, 30]
#j = 0 ; t = 0
#for col in range(4):
#    ax = fig1.add_subplot(spec1[t, col], projection=proj)
#    # Tick setting
#    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=25)
#    # Equator line
#    ax.plot([40,120],[0,0],'k--')
#
#    # Coast Line
#    ax.coastlines(resolution='110m', lw=1.75)
#    # Vector Map
#    q  =  ax.quiver(du['lon'].data, du['lat'].data, du.data[fig_number[j]], dv.data[fig_number[j]], 
#        regrid_shape=15, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
#        scale_units='xy', scale=1.1,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#        units='xy', width=0.25,
#        transform=proj,
#        color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)
#    # Add title of day
#    #ax.set_title(fig_number1[j], loc='left', fontsize=25)
#    #ax.set_title('Climatology', loc='right', fontsize=25)
#        
#    # Add the Figure number
#    # ax.set_title("("+number[j]+")",loc='left',fontsize=27.5)
#    # Add legend of the vector
#    #add_vector_legend(ax=ax,q=q, speed=5)
#    j+=1
#
#plt.savefig('test.png')