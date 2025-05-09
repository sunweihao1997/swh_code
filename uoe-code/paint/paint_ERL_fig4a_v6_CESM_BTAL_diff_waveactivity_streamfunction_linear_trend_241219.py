'''
2024-7-22
This script is to paint the difference between two periods in the streamfunction, meridional wind at 200 hPa

Linear trend
'''
import xarray as xr
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
sys.path.append('/home/sun/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# ================================ File location =========================================

path_src = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_src = 'Aerosol_Research_CESM_BTAL_BTALnEU_200hPa_streamfunction_velocity_potential.nc'

# ========================================================================================

file0  =  xr.open_dataset('/home/sun/data/download_data/data/wave_activity/BTAL_BTALnEU_diff_Z3_for_TN2001-Fx.monthly.1901_1955_trend.nc')
lat    =  file0.lat.data
lon    =  file0.lon.data

# =========== Calculate linear trend ==============
def calculate_linear_trend(start, end, input_array, varname):
    from scipy.stats import linregress

    print(input_array)
    time_dim, lat_dim, lon_dim = input_array.sel(time=slice(start, end), level=200)[varname].shape

    trend_data = np.zeros((lat_dim, lon_dim))
    p_data     = np.zeros((lat_dim, lon_dim))

    input_data = input_array.sel(time=slice(start, end), level=200)[varname].data
    #print(input_data.shape)

    for i in range(lat_dim):
        for j in range(lon_dim):
            #print(linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j]))
            slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j])
            trend_data[i, j] = slope
            p_data[i, j]    = p_value

    return trend_data, p_data

def calculate_linear_trend_diff(start, end, input_array, varname1, varname2):
    '''This calculate the linear trend difference between 2 given data'''
    from scipy.stats import linregress

    time_dim, lat_dim, lon_dim = input_array.sel(time=slice(start, end))[varname1].shape

    trend_data = np.zeros((lat_dim, lon_dim))
    p_data     = np.zeros((lat_dim, lon_dim))

    input_data1 = input_array.sel(time=slice(start, end))[varname1].data
    input_data2 = input_array.sel(time=slice(start, end))[varname2].data
    #print(input_data.shape)

    for i in range(lat_dim):
        for j in range(lon_dim):
            #print(linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j]))
            slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1, time_dim, time_dim), input_data1[:, i, j] - input_data2[:, i, j])
            trend_data[i, j] = slope
            p_data[i, j]    = p_value

    return trend_data, p_data

# ================================ Painting ==============================================

def paint_jjas_diff(sf, u, v, p, pic_name, left_title):
    '''This function paint the Diff aerosol JJA'''
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(20, 10), subplot_kw={'projection': proj})

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  0,140,10,65
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,135,10,dtype=int),yticks=np.linspace(0,80,9,dtype=int),nx=1,ny=1,labelsize=10.5)

    # Here I insert calculation about the zonal deviation of the stream function
    sf_d = sf.copy()
    for i in range(len(lat)):
        sf_d[i] = sf_d[i] - np.nanmean(sf[i])

    # contourf for the meridional wind v
    im1  =  ax.contourf(lon, lat, sf_d, levels=np.linspace(-10, 10, 11), cmap='coolwarm', alpha=1, extend='both')

    # contour for the streamfunction
#    im2  =  ax.contour(lon, lat, sf, levels=np.linspace(-2.5, 2.5, 11), alpha=1, colors='k',)
#    ax.clabel(im2, fontsize=5, inline=True)

    # stippling
    plt.rcParams.update({'hatch.color': 'gray'})
    #print(p.shape)
    #sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['//'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='110m', lw=1.5)

    q  =  ax.quiver(lon, lat, u, v, 
        regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
        scale_units='xy', scale=2.5     ,        # scale是参考矢量，所以取得越大画出来的箭头就越短
        units='xy', width=0.55,              # width控制粗细
        transform=proj,
        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

    add_vector_legend(ax=ax, q=q, speed=10, fontsize=12, location=(0.8, 0), length=0.225, quiver_x=0.9)


    #ax.set_ylabel("Influence of EU emission", fontsize=11)

    ax.set_title(left_title, loc='left',         fontsize=12)
    ax.set_title('Stream-function', loc='right', fontsize=12)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    plt.savefig('/home/sun/paint/ERL/{}'.format(pic_name))
    #plt.savefig('test.png', dpi=600)



def main():
    # 1. Firstly, calculate difference between two periods for each experiment for the streamfunction
    ncfile_fx   =  xr.open_dataset('/home/sun/data/download_data/data/wave_activity_corrected_test4/BTAL_TN2001-Fx.monthly.btal_minus_btalneu_1901_1955_linear_trend.nc').sel(level=300)
    ncfile_fy   =  xr.open_dataset('/home/sun/data/download_data/data/wave_activity_corrected_test4/BTAL_TN2001-Fy.monthly.btal_minus_btalneu_1901_1955_linear_trend.nc').sel(level=300)
    ncfile_psi  =  xr.open_dataset('/home/sun/data/download_data/data/wave_activity_corrected_test4/BTAL_psidev.monthly.period2.nc').sel(level=300)

#    p1 = 1901 ; p2 = 1955
#    x_con, x_p_con = calculate_linear_trend(p1, p2, ncfile_fx, 'Fx')
#    y_con, y_p_con = calculate_linear_trend(p1, p2, ncfile_fy, 'Fy')
#    z_con, z_p_con = calculate_linear_trend(p1, p2, ncfile_dv, 'div')
#
#    ncfile_fx   =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/wave_activity/BTALnEU_TN2001-Fx.monthly.157year_jjas.nc')
#    ncfile_fy   =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/wave_activity/BTALnEU_TN2001-Fy.monthly.157year_jjas.nc')
#    ncfile_dv   =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/wave_activity/BTALnEU_TN2001-Fz.monthly.157year_jjas.nc')
#
#    p1 = 1901 ; p2 = 1955
#    x_con_btalneu, x_p_con = calculate_linear_trend(p1, p2, ncfile_fx, 'Fx')
#    y_con_btalneu, y_p_con = calculate_linear_trend(p1, p2, ncfile_fy, 'Fy')
#    z_con_btalneu, z_p_con = calculate_linear_trend(p1, p2, ncfile_dv, 'div')


    #sys.exit()
    #print(np.nanmean(z_con))
    #sys.exit()
#    print(np.nanmean(ncfile_psi['psidev'].data))
#    sys.exit()

    paint_jjas_diff(ncfile_psi['psidev'].data*1e-5*55, 55 * (ncfile_fx['Fx'].data) * 1e4, 55 * (ncfile_fy['Fy'].data) * 1e4, None, "ERL_fig4a_v6_CESM_BTAL_wave_activity_linear_trend_300_vector_legend.pdf", '1901-1955')
    print("Paint Success")
#    paint_jjas_diff2(sf_diff/1e5, w_diff, None, "ERL_fig3_type2_rp_v_to_w_CESM_BTAL_streamfunction_meridional_wind_period_diff_150.pdf", '(a)')


if __name__ == '__main__':
    main()