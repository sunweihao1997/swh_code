'''
2025-5-23
This script is to plot the diagram of Fig. 12 in the article
'''
import xarray as xr
import numpy as np
import os
import sys
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from metpy.units import units
from matplotlib.path import Path
import matplotlib.patches as patches

sys.path.append('/home/sun/swh_code/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend


proj    =  ccrs.PlateCarree()
fig,ax   =  plt.subplots(figsize=(13,10),subplot_kw=dict(projection=ccrs.PlateCarree()))

# 范围设置
lonmin,lonmax,latmin,latmax  =  45,135,-10,30
extent     =  [lonmin,lonmax,latmin,latmax]


# 刻度设置
set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,130,5,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=20)
    
# 绘制赤道线
#ax.plot([40,150],[0,0],'--',color='k')
ax.coastlines(resolution='110m',lw=1.5)
ax.add_feature(cfeature.LAND, facecolor='whitesmoke')
#ax.add_feature(cfeature.OCEAN, facecolor='blue')

plt.savefig("/home/sun/paint/CD/Fig12_diagram.pdf", dpi=450)
