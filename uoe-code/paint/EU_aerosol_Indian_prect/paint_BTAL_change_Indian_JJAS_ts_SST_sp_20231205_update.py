'''
2023-12-05
This script modifed from other scripts and combine the calculation and paint, the involved variables include TS SST PS
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
sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# select two period to calculate difference
class period:
    ''' This class infludes the two periods for compare '''
    periodA_1 = 50 ; periodA_2 = 70
    periodB_1 = 90 ; periodB_2 = 110


class calculate_class:
    member_num = 8
    out_path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/'
    def cdo_ensemble(path1, exp_name, var_name, member_num):
        '''This function cdo the emsemble member result'''
        import os

        out_path1 = calculate_class.out_path0 + var_name + '/' + exp_name + '/'
        check_path(out_path1, 1)
        for i in range(member_num):
            path0 = path1 + exp_name + '/' + exp_name + '_' + str(i + 1) + '/' + 'mon/atm/' + var_name + '/'
            
            os.system('cdo cat ' + path0 + '*nc ' + out_path1 + exp_name + '_' + var_name + '_' + '1850_157years_member_' + str(i + 1) + '.nc')

    ensemble_path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/'
    def cal_ensemble(exp_name, var_name):
        '''This function calculate the ensemble mean among all the members'''
        ensemble_path1 = calculate_class.ensemble_path0 + var_name + '/' + exp_name + '/' # This path saves the ncfiles
        file_list = os.listdir(ensemble_path1)
        ref_file  = xr.open_dataset(ensemble_path1 + file_list[0])

        # === Claim the array for result ===
        var_2d = np.zeros((1883, 96, 144))

        for i in range(calculate_class.member_num):
            f0 = xr.open_dataset(ensemble_path1 + exp_name + '_' + var_name + '_' + '1850_157years_member_' + str(i + 1) +'.nc')

            var_2d += f0[var_name].data/calculate_class.member_num

        print('Successfully calculated result for {} {}'.format(exp_name, var_name))

        # Write to nc file
        ncfile  =  xr.Dataset(
            {
                var_name: (["time", "lat", "lon"], var_2d),
            },
            coords={
                "time": (["time"], ref_file['time'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
            },
            )

        ncfile[var_name].attrs = ref_file[var_name].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-20. This file is the ensemble mean among the 8 member in the {} emission experiments. The variable is {}.'.format(exp_name, var_name)
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/{}_{}_ensemble_mean_231020.nc".format(exp_name, var_name), format='NETCDF4')
        ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_slp_231020.py'

    def cal_jjas_mean(file_path, file_name, var_name, exp_name):
        '''This function calculate JJA mean for the data, which has been cdo cat and Ensemble_Mean'''
        file0 = xr.open_dataset(file_path + file_name)

        # === Claim 150 year array ===
        jjas_mean = np.zeros((157, 96, 144))

        data_JJAS = file0.sel(time=file0.time.dt.month.isin([6, 7, 8, 9]))

        for yyyy in range(157):
            jjas_mean[yyyy] = np.average(data_JJAS[var_name].data[yyyy * 4 : yyyy * 4 + 4], axis=0)
        
        print('{} {} JJAS mean calculation succeed!'.format(exp_name, var_name))

        # === Write to ncfile ===
        ncfile  =  xr.Dataset(
            {
                "{}_JJAS".format(var_name): (["time", "lat", "lon"], jjas_mean),
            },
            coords={
                "time": (["time"], np.linspace(1850, 1850+156, 157)),
                "lat":  (["lat"],  file0['lat'].data),
                "lon":  (["lon"],  file0['lon'].data),
            },
            )

        ncfile["{}_JJAS".format(var_name)].attrs = file0[var_name].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-20, modified on 2023-12-05. This file is JJAS mean for the {}. Experiment is {}.'.format(var_name, exp_name)
        ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_slp_231020_20231205_update.py'
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/{}_{}_ensemble_mean_JJAS_231020.nc".format(var_name, exp_name), format='NETCDF4')
        

    def cal_two_periods_difference(data):
        ref_file    = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/model_data/WIND/BTAL/U/BTAL_U_1850_150years_member_1.nc')

        avg_periodA = np.average(data[period.periodA_1 : period.periodA_2], axis=0)
        avg_periodB = np.average(data[period.periodB_1 : period.periodB_2], axis=0)

        # Add coordination to the array
        ncfile  =  xr.Dataset(
            {
                "jja_diff": (["lev", "lat", "lon"], avg_periodB - avg_periodA),
            },
            coords={
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
                "lev":  (["lev"],  ref_file['lev'].data),
            },
            )
        
        return ncfile

def paint_jjas_diff_BTAL(sst, ncfile):
    lon = ncfile.lon.data
    lat = ncfile.lat.data

    '''This function paint the Diff aerosol JJA'''
    proj       =  ccrs.PlateCarree()
    ax         =  plt.subplot(projection=proj)

    # Create the subplot
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    # Tick setting
    # extent
    lonmin,lonmax,latmin,latmax  =  30,110,-10,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(0,140,8,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=10.5)

    # contourf for the geopotential height
    im1  =  ax.contourf(lon, lat, sst, levels=np.linspace(-0.5, 0.5, 21), cmap='coolwarm', alpha=1, extend='both')

    # stippling
    #plt.rcParams.update({'hatch.color': 'gray'})
    #sp  =  ax.contourf(lon, lat, p, levels=[0.1, 1], colors='none', hatches=['...'])

    # contour for the meridional wind
    #im2  =  ax.contour(lon, lat, z, 6, colors='green')
    #ax.clabel(im2, inline=True, fontsize=10)

    ax.coastlines(resolution='110m', lw=1.1)
    ax.add_feature(cartopy.feature.LAND, zorder=100, facecolor='white')
    #ax.set_global()

    #ax.set_ylabel("BTAL long-term changes", fontsize=11)

    #ax.set_title('1901-1920 to 1941-1960', fontsize=12.5)
    ax.set_title('BTAL', loc='right', fontsize=12.5)

    # Add colorbar
    plt.colorbar(im1, orientation='horizontal')

    #plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/EUI_BTAL_OMEGA_diff_1900_1960.pdf')
    plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/circulation/BTAL_diff_JJAS_SST.pdf')

def main():
    # === 1. Archive the intepolated modei output === !! Finished 2023-10-18-19:54!!
    #archive_interpolated_model_output()

    # === 2. Cdo cat the long-term files === !! Finished 2023-10-20-10:14!!
#    src_path = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/'
#    calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTAL', var_name='PS', member_num=8)
#    calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTALnEU', var_name='PS', member_num=8)
#    calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTAL', var_name='SST', member_num=8)
#    calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTALnEU', var_name='SST', member_num=8)
#    calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTAL', var_name='TS', member_num=8)
#    calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTALnEU', var_name='TS', member_num=8)
#
#    print('TS PS SST CDO ensemble has completed!')
#
#
#    # === 3. Calculate ensemble mean among the members output === !! Finished 2023-10-20-10:21!!
#    calculate_class.cal_ensemble('BTAL', 'PS')
#    calculate_class.cal_ensemble('BTALnEU', 'PS')
#    calculate_class.cal_ensemble('BTAL', 'SST')
#    calculate_class.cal_ensemble('BTALnEU', 'SST')
#    calculate_class.cal_ensemble('BTAL', 'TS')
#    calculate_class.cal_ensemble('BTALnEU', 'TS')
#
#    print('TS PS SST ensemble-mean has completed!')
#
#    # === 4. Calculate JJAS mean === !!Finished 2023-10-20-10:26!! 
#    analysis_data_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
#    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
#                                file_name='BTAL_PS_ensemble_mean_231020.nc', 
#                                var_name='PS',
#                                exp_name='BTAL',
#                                )
#    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
#                                file_name='BTALnEU_PS_ensemble_mean_231020.nc', 
#                                var_name='PS',
#                                exp_name='BTALnEU',
#                                )
#    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
#                                file_name='BTAL_TS_ensemble_mean_231020.nc', 
#                                var_name='TS',
#                                exp_name='BTAL',
#                                )
#    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
#                                file_name='BTALnEU_TS_ensemble_mean_231020.nc', 
#                                var_name='TS',
#                                exp_name='BTALnEU',
#                                )
#    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
#                                file_name='BTAL_SST_ensemble_mean_231020.nc', 
#                                var_name='SST',
#                                exp_name='BTAL',
#                                )
#    calculate_class.cal_jjas_mean(file_path=analysis_data_path, 
#                                file_name='BTALnEU_SST_ensemble_mean_231020.nc', 
#                                var_name='SST',
#                                exp_name='BTALnEU',
#                                )
#
#    print('TS PS SST ensemble-mean JJAS has completed!')
    # === Step 4 !!Finished 2023-10-19:14:26!!===

    # === 5. Calculate difference between two given period ===
    file_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'

    periodA_1 = 1900 ; periodA_2 = 1920
    periodB_1 = 1940 ; periodA_2 = 1960

    SST_BTAL    = xr.open_dataset(file_path + "SST_BTAL_ensemble_mean_JJAS_231020.nc")
    SST_BTALnEU = xr.open_dataset(file_path + "SST_BTALnEU_ensemble_mean_JJAS_231020.nc")

    SST_BTAL_1  = SST_BTAL.sel(time=slice(1900, 1920)) ; SST_BTAL_2  =  SST_BTAL.sel(time=slice(1940, 1960))
    SST_BTALnEU_1  = SST_BTALnEU.sel(time=slice(1900, 1920)) ; SST_BTALnEU_2  =  SST_BTALnEU.sel(time=slice(1940, 1960))

    SST_diff_BTAL    = np.average(SST_BTAL_2['SST_JJAS'].data, axis=0) - np.average(SST_BTAL_1['SST_JJAS'].data, axis=0)
    SST_diff_BTALnEU = np.average(SST_BTALnEU_2['SST_JJAS'].data, axis=0) - np.average(SST_BTALnEU_1['SST_JJAS'].data, axis=0)

    SST_diff         = SST_diff_BTAL - SST_diff_BTALnEU # Influence of the EU emission

    paint_jjas_diff_BTAL(SST_diff_BTAL, SST_BTAL)

#    
#    
#    # ================= 6. Painting section =========================================
#    # === 6.1 Period difference among the control experiment and noEU experiment themsleves ===
#    paint_class.plot_uv_diff_at_select_lev(con_diff_u, con_diff_v, 850, plot_name='BTAL')
#    paint_class.plot_uv_diff_at_select_lev(eu_diff_u,  eu_diff_v,  850, plot_name='BTALnEU')
#
#    paint_class.plot_uv_diff_at_select_lev_among_exps(con_diff_u,  con_diff_v, eu_diff_u, eu_diff_v, 850,)
#    #print(np.nanmin(con_diff_u['jja_diff']))



if __name__ == '__main__':
    main()