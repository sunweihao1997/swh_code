'''
2024-5-27
This script is to plot the changes in MJJAS ts between SSP370 and SSP370lowNTCF, the simulation of historical is ignored
'''
import xarray as xr
import numpy as np
from scipy import stats
import cartopy.util as cutil

path_in   =  '/home/sun/data/process/analysis/AerChem/'

models_label = ['EC-Earth3-AerChem', 'UKESM1-0-LL', 'GFDL-ESM4', 'MRI-ESM2','MIROC6',] # GISS provide no daily data
#models_label = ['EC-Earth3-AerChem','UKESM1-0-LL', 'GFDL-ESM4','MRI-ESM2'] # GISS provide no daily data
#models_label = ['MPI-ESM-1-2-HAM', 'MIROC6', 'GISS-E2-1-G'] # GISS provide no daily data

varname      = 'div'

gen_f     = xr.open_dataset('/home/sun/data/topography/geo_1279l4_0.1x0.1.grib2_v4_unpack.nc')

z         = gen_f['z'].data[0] / 9.8

def cal_multiple_model_avg(f0, exp_tag, timeaxis,):
    '''
    Because the input data is single model, so this function is to calculate the model-averaged data

    timeaxis is 65 for historical and 36 for furture simulation
    '''
    # 1. Generate the averaged array
    lat = f0.lat.data ; lon = f0.lon.data ; time = f0[timeaxis].data ; 

    multiple_model_avg = np.zeros((len(time), len(lat), len(lon)))

    # 2. Calculation
    models_num = len(models_label)

    for mm in models_label:
        varname1 = mm + '_' + exp_tag

        multiple_model_avg += (f0[varname1].data / models_num)

    #
    return multiple_model_avg

def plot_change_wet_day(ssp, sspntcf, omega, left_string, figname, lon, lat, parray, ct_level=np.linspace(-10., 10., 21),):
    '''
    This function is to plot the changes in the wet day among the SSP370 and SSP370lowNTCF

    This figure contains three subplot: 1. changes between SSP370 and historical 2. changes between SSP370lowNTCF and historical 3. NTCF mitigation (ssp370 - ssp370lowNTCF)
    '''
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    import sys
    sys.path.append("/home/sun/local_code/module/")
    from module_sun import set_cartopy_tick

    # -------   cartopy extent  -----
    lonmin,lonmax,latmin,latmax  =  40,130,0,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # -------     figure    -----------
    #proj  =    ccrs.Robinson(central_longitude=180)
    #proj   =    ccrs.PlateCarree(central_longitude=180)
    fig1    =  plt.figure(figsize=(10,8))
    spec1   =  fig1.add_gridspec(nrows=1,ncols=1)

    left_title = '{}'.format(left_string)
    right_title= ['SSP370 - SSP370lowNTCF']

    pet        = [(ssp-sspntcf)]

    # -------    colormap --------------
    coolwarm = plt.get_cmap('coolwarm')

    colors = [
    (coolwarm(0.0)),  # 对应level -3
    (coolwarm(0.25)), # 对应level -2
    (1, 1, 1, 1),     # 插入白色，对应level -1到1
    (coolwarm(0.75)), # 对应level 2
    (coolwarm(1.0))   # 对应level 3
    ]

    new_cmap = LinearSegmentedColormap.from_list('custom_coolwarm', colors)

    # ------      paint    -----------
    for row in range(1):
        col = 0
        ax = fig1.add_subplot(spec1[row,col], projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_global()
        

        # 设置刻度
        set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(40,130,7,dtype=int),yticks=np.linspace(0,50,6,dtype=int),nx=1,ny=1,labelsize=15)

        # 添加赤道线
        #ax.plot([40,150],[0,0],'k--')

        # add cyclic point
#        lon2d, lat2d = np.meshgrid(lon, lat)
#        cdata, clon2d, clat2d = cutil.add_cyclic(pet[row], lon2d, lat2d)
#
#        im  =  ax.contourf(clon2d, clat2d, cdata, ct_level, cmap=new_cmap, alpha=1, extend='both')
        im  =  ax.contour(lon, lat, pet[row], ct_level, alpha=1, transform=ccrs.PlateCarree(), colors='k')
        ax.clabel(im, fontsize=9, inline=1)

        im2  =  ax.contourf(lon, lat, omega, np.linspace(-5, 5, 11), cmap='coolwarm', alpha=1, extend='both', transform=ccrs.PlateCarree(), )

        # t 检验
        sp  =  ax.contourf(lon, lat, parray, levels=[0., 0.05], colors='none', hatches=['.'], transform=ccrs.PlateCarree())

        # 海岸线
        ax.coastlines(resolution='50m',lw=1.65)

        #topo  =  ax.contour(gen_f['longitude'].data, gen_f['latitude'].data, z, levels=[3000], colors='brown', linewidths=3)

        ax.set_title(left_title, loc='left', fontsize=16.5)
        ax.set_title(right_title[row], loc='right', fontsize=16.5)


    # 加colorbar
    fig1.subplots_adjust(top=0.8) 
    cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig1.colorbar(im2, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=10)

    cb.set_ticks(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4,]))
    cb.set_ticklabels(np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4,]))

    plt.savefig("/home/sun/paint/AerChemMIP/AerChemMIP_modelgroup_spatial_MJJAS_ssp370_ssp370ntcf_{}.pdf".format(figname))

def cal_student_ttest(array1, array2):
    '''
        This function is to calculate the student ttest among the array1 and array2
    '''
    p_value = np.zeros((array1.shape[1], array2.shape[2]))

    for i in range(array1.shape[1]):
        for j in range(array2.shape[2]):
            #print(i)
            p_value[i, j] = stats.ttest_rel(array1[:, i, j], array2[:, i, j])[1]

    return p_value

if __name__ == '__main__':
    f0  =  xr.open_dataset('/home/sun/data/AerChemMIP/process/multiple_model_climate_ua_month_MJJAS.nc').sel(plev=20000)

    ssp0      =  cal_multiple_model_avg(f0, 'ssp',  'time_ssp')
    ntcf0     =  cal_multiple_model_avg(f0, 'sspntcf', 'time_ssp')

    ttest     =  cal_student_ttest(ssp0, ntcf0)

    f1  =  xr.open_dataset('/home/sun/data/AerChemMIP/process/multiple_model_climate_wap_month_MJJAS.nc').sel(plev=50000)

    omega     =  cal_multiple_model_avg(f1, 'ssp',  'time_ssp') - cal_multiple_model_avg(f1, 'sspntcf', 'time_ssp')

    ttest     =  cal_student_ttest(cal_multiple_model_avg(f1, 'ssp',  'time_ssp'), cal_multiple_model_avg(f1, 'sspntcf', 'time_ssp'))

    #print(np.nanmax(ssp0))

#    # Wave activity flux
#    f1  =  xr.open_dataset("/home/sun/data/AerChemMIP/process/AerChemMIP_SSP370_SSP370lowNTCF_diff_Z3_for_TN2001-Fx.MJJAS.nc").sel(level=300)
#    f2  =  xr.open_dataset("/home/sun/data/AerChemMIP/process/AerChemMIP_SSP370_SSP370lowNTCF_diff_Z3_for_TN2001-Fy.MJJAS.nc").sel(level=300)
#
#    print(np.nanmean(f1.Fx.data))



    # Note that the variable in the above is three-dimension while the first is the number os the year
    #levels    =  [-14, -12, -10, -8, -6, -4, -2, -1, 1, 2, 4, 6, 8, 10, 12, 14]
    #levels    =  [-1.0, -.8, -.6, -.4, -.2, -0.1, -.05, .05, 0.1, .2, .4, .6, .8, 1.0,]

    #levels    =  np.array([-8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8])
    levels    =  np.array([-12, -9, -6, -3, -1, 1, 3, 6, 9, 12])
    levels    =  np.linspace(-10, 10, 11)

    #plot_change_wet_day(np.nanmean(ssp0, axis=0) * 10e2, np.nanmean(ntcf0, axis=0) * 10e2, '200hPa v (MJJAS)', '200hPa v (MJJAS)', f0.lon.data, f0.lat.data, ttest, levels)
    plot_change_wet_day(-1*np.nanmean(ssp0[-20:], axis=0), -1*np.nanmean(ntcf0[-20:], axis=0), -1*np.nanmean(omega[-20:], axis=0)*1e3, '200hPa u (MJJAS)', '200hPa u (MJJAS)', f0.lon.data, f0.lat.data, ttest, levels/10)