'''
2023-10-16
This script calculate and paint aerosol changes between given 2 period, area is EU

data_path = /exports/csce/datastore/geos/users/s2618078/data/model_data/SO4/BURDENSO4_1850_2006.nc
The data is from CESM1 output

Basic information about BURDENSO4:
max is 0.00010, unit kg m^-2 
1 kg = 1000000
data includens nan

Focus on JJA
'''
import xarray as xr
import numpy as np
import os
from matplotlib import cm
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
sys.path.append('/home/sun/uoe-code/module/')
from module_sun import set_cartopy_tick
src_file  =  xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BURDENSO4_ensemble_mean_JJA_231016.nc')

lat = src_file.lat.data ; lon = src_file.lon.data
# select two period to calculate difference
#print(np.nanmax(src_file['BURDENSO4'].data))
class period:
    ''' This class infludes the two periods for compare '''
    periodA_1 = 50 ; periodA_2 = 70
    periodB_1 = 90 ; periodB_2 = 110

def plot_diff_slp_wind(diff_slp, left_title, right_title, out_path, pic_name, level, pvalue):
    '''This function plot the difference in precipitation'''

    # ------------ colormap ----------------------------------
#    cmap = cmr.prinsenvlag

    # ------------ level -------------------------------------

    levels = level

    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 7), subplot_kw={'projection': proj})

    # Tick settings
    cyclic_data_vint, cyclic_lon = add_cyclic_point(diff_slp, coord=lon)
    cyclic_data_p,    cyclic_lon = add_cyclic_point(pvalue, coord=lon)


    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  -20, 50, 25, 70
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(-15, 45, 5,dtype=int), yticks=np.linspace(20, 70, 6, dtype=int),nx=1,ny=1,labelsize=15.5)

    # Shading for SLP difference
    im   =  ax.contourf(cyclic_lon, lat, cyclic_data_vint, levels=levels, cmap='coolwarm', alpha=1, extend='both')
    dot  =  ax.contourf(cyclic_lon, lat, cyclic_data_p, levels=[0., 0.1], colors='none', hatches=['/'])

    
    # Vectors for Wind difference
#    q  =  ax.quiver(lon, lat, diff_u, diff_v, 
#                        regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
#                        scale_units='xy', scale=1.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#                        units='xy', width=0.35,              # width控制粗细
#                        transform=proj,
#                        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)
    
    #add_vector_legend(ax=ax, q=q, speed=5)

#    # Stippling picture
    #sp  =  ax.contourf(lon, lat, p_value, levels=[0., 0.1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='110m', lw=1.5)

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=15.5)
    ax.set_title(right_title, loc='right', fontsize=15.5)

    # ========= add colorbar =================
    cb  =  fig.colorbar(im, shrink=0.5, pad=0.1, orientation='horizontal')


    plt.savefig(out_path + pic_name)

class cal_function:
    ''' This Class includes the function for calculation '''
    def cal_seasonal_mean_in_given_months(month, data):
        '''This function calculate seasonal mean, the character is the first month of the season'''

        # === Claim the array for saving data ===
        smean = np.zeros((150, 96, 144))

        for yyyy in range(150):
            smean[yyyy] = np.nanmean(data[yyyy * 12 + month - 1 : yyyy * 12 + month + 2,], axis=0) # The time axis is month

        return smean

    def write_file(data):
        ncfile  =  xr.Dataset(
        {
            "BURDENSO4_JJA": (["time", "lat", "lon"], data),
        },
        coords={
            "time": (["time"], np.linspace(1850, 1850+149, 150)),
            "lat":  (["lat"],  src_file['lat'].data),
            "lon":  (["lon"],  src_file['lon'].data),
        },
        )

        ncfile['BURDENSO4_JJA'].attrs = src_file['BURDENSO4'].attrs
        ncfile.attrs['description']  =  'Created on 2023-10-16. This file is the JJA ensemble mean among the 8 member in the BTAL emission experiments. The variables are BURDENSO4.'
        ncfile.to_netcdf("/home/sun/data/download_data/data/analysis_data/BTAL_sulfate_column_burden_jja_mean_231016.nc", format='NETCDF4')

    out_path = '/exports/csce/datastore/geos/users/s2618078/data/model_data/BURDENSO4/'
    member_num = 8
    def cdo_ensemble(path, exp_name, var_name, member_num):
        '''This function cdo the emsemble member result'''
        import os

        for i in range(member_num):
            path_src = path + exp_name + '_' + str(i + 1) + '/mon/atm/' + var_name + '/'
            
            os.system('cdo cat ' + path_src + '*nc ' + cal_function.out_path + 'BTAL_BURDENSO4_1850_150years_member_' + str(i + 1) + '.nc')
    
    def cal_ensemble(var_name):
        '''This function calculate the ensemble mean among all the members'''
        file_list = os.listdir(cal_function.out_path)
        ref_file  = xr.open_dataset(cal_function.out_path + file_list[0])

        # === Claim the array for result ===
        so4 = np.zeros((1883, 96, 144))

        for i in range(cal_function.member_num):
            f0 = xr.open_dataset(cal_function.out_path + 'BTAL_BURDENSO4_1850_150years_member_' + str(i + 1) +'.nc')

            so4 += f0['BURDENSO4'].data/cal_function.member_num

        print('Successfully calculated result')

        # Write to nc file
        ncfile  =  xr.Dataset(
            {
                "BURDENSO4": (["time", "lat", "lon"], so4),
            },
            coords={
                "time": (["time"], ref_file['time'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
            },
            )

        ncfile['BURDENSO4'].attrs = ref_file['BURDENSO4'].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-16. This file is the ensemble mean among the 8 member in the BTAL emission experiments. The variable is sulfate column burden.'
        ncfile.to_netcdf("/home/sun/data/download_data/data/analysis_data/BTAL_BURDENSO4_ensemble_mean_231016.nc", format='NETCDF4')

    file_name  = '/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BURDENSO4_ensemble_mean_231016.nc'
    def cal_jja_mean():
        '''This function calculate JJA mean for the data, which has been cdo cat and Ensemble_Mean'''
        file0 = xr.open_dataset(cal_function.file_name)

        file0 = file0.sel(time=file0.time.dt.month.isin([6, 7, 8,]))

        # === Claim 150 year array ===
        jja_mean = np.zeros((150, 96, 144))

        for yyyy in range(150):
            jja_mean[yyyy] = np.average(file0['BURDENSO4'].data[yyyy * 3 : yyyy * 3 + 3], axis=0)
        
        print('JJA mean calculation succeed!')

        # === Write to ncfile ===
        ncfile  =  xr.Dataset(
            {
                "BURDENSO4_JJA": (["time", "lat", "lon"], jja_mean),
            },
            coords={
                "time": (["time"], np.linspace(1850, 1999, 150)),
                "lat":  (["lat"],  file0['lat'].data),
                "lon":  (["lon"],  file0['lon'].data),
            },
            )

        ncfile['BURDENSO4_JJA'].attrs = file0['BURDENSO4'].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-16. This file is JJA mean for the ensemble-mean sulfate column burden. The variable is sulfate column burden.'
        ncfile.to_netcdf("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BURDENSO4_ensemble_mean_JJA_240115.nc", format='NETCDF4')

    def cal_two_periods_difference(data):
        avg_periodA = np.average(data[period.periodA_1 : period.periodA_2], axis=0)
        avg_periodB = np.average(data[period.periodB_1 : period.periodB_2], axis=0)

        return avg_periodB - avg_periodA

class plot_function:
    '''This class save the settings and function for painting'''
    # ======== Set Extent ==========
    lonmin,lonmax,latmin,latmax  =  -30,150,0,80
    extent     =  [lonmin,lonmax,latmin,latmax]

    # ======== Set Figure ==========
    proj       =  ccrs.PlateCarree()

    viridis = cm.get_cmap('Reds', 11)
    newcolors = viridis(np.linspace(0, 1, 11))
    newcmp = ListedColormap(newcolors)
    newcmp.set_under('white')

    def paint_jja_diff(data):
        '''This function paint the Diff aerosol JJA'''
        proj       =  ccrs.PlateCarree()
        ax         =  plt.subplot(projection=proj)

        # Tick setting
        cyclic_data, cyclic_lon = add_cyclic_point(data, coord=src_file['lon'].data)
        set_cartopy_tick(ax=ax,extent=plot_function.extent,xticks=np.linspace(-30,150,7,dtype=int),yticks=np.linspace(0,80,5,dtype=int),nx=1,ny=1,labelsize=12)

        im2  =  ax.contourf(cyclic_lon, src_file['lat'].data, cyclic_data * 10e6, np.linspace(30,130,6), cmap=plot_function.newcmp, alpha=1, extend='both')

        ax.coastlines(resolution='50m', lw=1.2)

        #bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none', alpha=0.7)
        #ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

        #ax.set_title('1901-1921 to 1941-1961', fontsize=15)

        # Add colorbar
        plt.colorbar(im2, orientation='horizontal')

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/aerosol/diff_JJA_EU_aerosol.pdf')

    def paint_time_series_model(data_0,):
        '''
            This function plot the time-series for the 1-D data
            data0: Raw data
            data1: moving-average data
            w    : moving parameter
        '''
        fig, ax = plt.subplots()

        ax.plot(np.linspace(1891, 1891 + 109, 109), data_0 , color='orange', linestyle='solid', linewidth=2.5)

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/aerosol/' + 'model_aerosol_long_term_series.pdf')

def calculate_linear_trend(start, end, input_array, varname):
    from scipy.stats import linregress

    time_dim, lat_dim, lon_dim = input_array.sel(time=slice(start, end))[varname].shape

    trend_data = np.zeros((lat_dim, lon_dim))
    p_data     = np.zeros((lat_dim, lon_dim))

    input_data = input_array.sel(time=slice(start, end))[varname].data
    #print(input_data.shape)

    for i in range(lat_dim):
        for j in range(lon_dim):
            #print(linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j]))
            slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j])
            trend_data[i, j] = slope
            p_data[i, j]    = p_value

    return trend_data, p_data

def main():
    # === 1. CDO emsemble members files === !!Finished!!
    # cal_function.cdo_ensemble('/exports/csce/datastore/geos/groups/aerosol/sabineCESM/BTAL/', 'BTAL', 'BURDENSO4', 8)

    # === 2. Calculate ensemble mean among the members output === !! Finished !!
    #cal_function.cal_ensemble('BURDENSO4')

    # === 3. Calculate JJA mean === !!Finished!! /home/sun/data/download_data/data/analysis_data/BTAL_BURDENSO4_ensemble_mean_JJA_231016.nc
    cal_function.cal_jja_mean()

    # === 4. Calculate difference between two given period ===
    file0 = xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BURDENSO4_ensemble_mean_JJA_240115.nc')
    
    p1 = 1901 ; p2 = 1955
    psl_con, psl_p_con = calculate_linear_trend(p1, p2, file0, 'BURDENSO4_JJA')

    #sys.exit('Complete!')
    #psl_neu, psl_p_neu = calculate_linear_trend(p1, p2, psl_btalneu, 'FSDSCS_JJA')
    
    #jja_mean_diff = cal_function.cal_two_periods_difference(file0['BURDENSO4_JJA'].data)
    #print('Successfully calculated JJA difference')
    #print(np.nanmin(jja_mean_diff))
    #print(np.nanmax(jja_mean_diff))

    # === 5. Write calculation result to ncfile ===
    ncfile  =  xr.Dataset(
        {
            "BURDENSO4_JJA": (["lat", "lon"], psl_con),
            "BURDENSO4_JJA_p": (["lat", "lon"], psl_p_con),
        },
        coords={
            "lat":  (["lat"],  src_file['lat'].data),
            "lon":  (["lon"],  src_file['lon'].data),
        },
        )
##
    ncfile['BURDENSO4_JJA'].attrs = src_file['BURDENSO4_JJA'].attrs
    ncfile.attrs['description']  =  'Created on 2023-10-17. This file is the difference for period (1900-1920) and (1920-1960) JJA ensemble mean among the 8 member in the BTAL emission experiments. The variables are BURDENSO4_JJA.'
    ncfile.to_netcdf("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_sulfate_column_burden_jja_mean_linear_trend_1901_1955_241002.nc", format='NETCDF4')
    # === 5 Step Succeed ===

    # === 6. painting ===
    file1 = xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_sulfate_column_burden_jja_mean_linear_trend_1901_1955_241002.nc')



    out_path  = "/home/sun/paint/ERL/"
    level2    =  np.linspace(-20, 20, 11)
    print(np.nanmean(psl_con))
    plot_diff_slp_wind(diff_slp=1e8*gaussian_filter((psl_con), sigma=0.35),left_title='1901-1955 Linear Trend', right_title='CESM_ALL - CESM_noEU', out_path=out_path, pic_name="ERL_fig3a_v5_JJA_BTAL_BTALnEU_SO4_linear_trend.pdf", level=level2, pvalue=psl_p_con)

#    # === 7. paint EU-area aerosol time series ===
#    file2 = xr.open_dataset('/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BURDENSO4_ensemble_mean_JJA_231016.nc').sel(time=slice(1891, 1999), lat=slice(30, 60), lon=slice(5, 35))
#
#    EU_mean = np.zeros((109))
#    for yyyy in range(109):
#        EU_mean[yyyy] = np.average(file2['BURDENSO4_JJA'].data[yyyy])
#
#    plot_function.paint_time_series_model(EU_mean * 10e6)
    


    

    

if __name__ == '__main__':
    main()