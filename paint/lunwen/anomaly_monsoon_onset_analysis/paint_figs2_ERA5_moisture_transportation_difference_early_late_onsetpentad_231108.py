'''
2023-11-8
This script is to see the difference in the moisture transportation between onset early/late years

The goal is to support the argument that Somali CEF induce more water transportation to the BOB area
This script also includes the calculation process
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


f0  =  xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/moisture_transportation_and_vertical_integration_composite_abnormal.nc").sel(lev=850)

# Due to the vertical integration do not have coordination, here I add it
ncfile  =  xr.Dataset(
{
    "mt_vint_early": (["time", "lat", "lon"], f0['mt_vint_early'].data),
    "mt_vint_late":  (["time", "lat", "lon"], f0['mt_vint_late'].data),
},
coords={
    "time": (["time"], f0['time'].data),
    "lat":  (["lat"],  f0['lat'].data),
    "lon":  (["lon"],  f0['lon'].data),
},
)

# Select the onset date
f0_onset     = f0.sel(time=slice(0, 5))
ncfile_onset = ncfile.sel(time=slice(0, 5))

# Calculate difference
u_diff       = np.average(f0_onset['mtu_late'].data, axis=0) - np.average(f0_onset['mtu_early'].data, axis=0)
v_diff       = np.average(f0_onset['mtv_late'].data, axis=0) - np.average(f0_onset['mtv_early'].data, axis=0)
mt_diff      = np.average(ncfile_onset['mt_vint_late'].data, axis=0) - np.average(ncfile_onset['mt_vint_early'].data, axis=0)

# ========== Settings ===============
    # cmap use cmasher
cmap = cmr.waterlily_r
# levels
level = np.linspace(-200, 200, 11)
# extent
lonmin,lonmax,latmin,latmax  =  45,115,-10,30
extent     =  [lonmin,lonmax,latmin,latmax]

# ========== Painting ===============
proj         =  ccrs.PlateCarree()
fig1, ax    =  plt.subplots(figsize=(15,12),  subplot_kw={'projection': ccrs.PlateCarree()})



# Tick setting
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,30,5,dtype=int),nx=1,ny=1,labelsize=22)

# Equator line
ax.plot([40,120],[0,0],'k--')

# Shading for precipitation
#im  =  ax.contourf(f0.lon.data, f0.lat.data, mt_diff * 1e5, levels=level, cmap=cmap, alpha=1, extend='both')
im  =  ax.contourf(f0.lon.data, f0.lat.data, mt_diff * 1e5, levels=level, cmap=cmap, alpha=1, extend='both')
#im  =  ax.contourf(f0.lon.data, f0.lat.data, np.average(ncfile['mt_vint_early'].data[25:30], axis=0) * 1e5, cmap=cmap, alpha=1, extend='both')

# Coast Line
ax.coastlines(resolution='110m', lw=1.75)

q  =  ax.quiver(f0['lon'].data, f0['lat'].data, u_diff, v_diff, 
        regrid_shape=12, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=1.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.45,
        transform=proj,
        color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)

add_vector_legend(ax=ax,q=q, speed=5)
#q  =  ax.quiver(f0['lon'].data, f0['lat'].data, np.average(f0['mtu_early'].data[25:30], axis=0), np.average(f0['mtv_early'].data[25:30], axis=0), 
#        regrid_shape=12, angles='uv',   # regrid_shape这个参数越小，是两门就越稀疏
#        scale_units='xy', scale=1.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#        units='xy', width=0.25,
#        transform=proj,
#        color='k',linewidth=1.2,headlength = 5, headaxislength = 4, headwidth = 5,alpha=0.8)


# 3. Color Bar
fig1.subplots_adjust(top=0.8) 
cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
cb  =  fig1.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
cb.ax.tick_params(labelsize=25)

plt.savefig('/home/sun/paint/lunwen/anomoly_analysis/v0_figs2_water_transportation_diff.pdf', dpi=350)