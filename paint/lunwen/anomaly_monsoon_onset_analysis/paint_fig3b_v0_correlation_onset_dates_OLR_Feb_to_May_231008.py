'''
2023-10-8
This script paint the correlation between monsoon onset dates and the SST gradient
This figure serves for the Fig3a in my essay

Reference:

'''
from matplotlib import projections
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cartopy

sys.path.append("/home/sun/swh_code/module/")
from module_sun import *

sys.path.append("/home/sun/swh_code/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import set_cartopy_tick,save_fig
from paint_lunwen_version3_0_fig2a_tem_gradient_20220426 import add_text

def calculate_threshold_correlation(n=63):
    '''This function calculate the threshold for the given sample number
    list is here: https://wenku.baidu.com/view/fdfece05a6c30c2259019eed.html?_wkts_=1678086714615
    '''
    return 0.20912, 0.24803, 0.32227 # 0.1, 0.05, 0.01

corr_file = xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/index/correlation/onset_dates_with_OLR.nc')

corr_list = [corr_file['onset_with_olr'].data[1], corr_file['onset_with_olr'].data[2], corr_file['onset_with_olr'].data[3], corr_file['onset_with_olr'].data[4],]

month_name = ['Feb', 'Mar', 'Apr', 'May']
#month_name = ['(e)', '(f)', '(g)', '(h)']

def paint_12month_correlation(correlation, extent, lon, lat, title):
    '''This function plot the correlation among the 12 months'''
    from cartopy.util import add_cyclic_point
    # Figure Setting
    proj    =  ccrs.PlateCarree(central_longitude=180)
    fig1    =  plt.figure(figsize=(22.5,12))
    spec1   =  fig1.add_gridspec(nrows=1,ncols=4)

    # colorbar使用ncl的
    #cmap  =  create_ncl_colormap("/home/sun/data/color_rgb/MPL_coolwarm.txt",22)

    j = 0
    for col in range(4):
        for row in range(1):
            ax = fig1.add_subplot(spec1[row,col],projection=proj)

            # Set the tick and tick-label
            set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(40,140,6,dtype=int),yticks=np.linspace(30,-10,5,dtype=int),nx=1,ny=1,labelsize=12.5)

            # contourf
            #im = ax.contourf(lon, lat, correlation[j], [0, 0.24, 1], hatches=[None, '.'])
            #print(np.nanmax(correlation[j]))
            data = correlation[j]
            data, lons = add_cyclic_point(correlation[j], coord=lon)
            im = ax.contourf(lons, lat, data, np.linspace(-0.8, 0.8, 9), cmap='coolwarm', transform = ccrs.PlateCarree(), extend='both')
            #im = ax.contourf(lon, lat, data, np.linspace(-0.8, 0.8, 9), cmap=cmap,)
            im2 = ax.contourf(lons, lat, data, [-1, -0.3, 0, 0.3, 1], hatches=['..', None, None, '..'], colors="none", transform=ccrs.PlateCarree())
            ax.coastlines()
            

            ax.plot([0,360],[0,0],'k--',linewidth=0.5,transform = ccrs.PlateCarree())
           # ax.plot([30,330],[10,10],'r--',transform = ccrs.PlateCarree())

            ax.set_title(month_name[j], loc='right', fontsize=16)
            #ax.set_title(title, loc='right', fontsize=15.5)


            j+=1

    # 加colorbar
    fig1.subplots_adjust(top=0.8) 
    cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig1.colorbar(im, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=15)

    save_fig(path_out="/home/sun/paint/",file_out="phd_c5_v0_fig12_correlation_onsetdates_OLR.pdf")

def main():
    lonmin,lonmax,latmin,latmax  =  40,150,30,-10
    extent     =  [lonmin,lonmax,latmin,latmax]
    paint_12month_correlation(corr_list, extent, corr_file['lon'], corr_file['lat'], "OLR (99%)")

if __name__ == '__main__':
    main()