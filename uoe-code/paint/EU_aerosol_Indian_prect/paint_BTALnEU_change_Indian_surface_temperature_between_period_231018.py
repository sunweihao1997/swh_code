'''
2023-10-17
This script calculate and paint surface temperature changes between given 2 period, area is Indian

for the difference value: min -1.45 max 2.07
Focus on JJA
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

src_file  =  xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTALnEU_TS_ensemble_mean_231018.nc')

# select two period to calculate difference
#print(np.nanmax(src_file['TS'].data))
class period:
    ''' This class infludes the two periods for compare '''
    periodA_1 = 50 ; periodA_2 = 70
    periodB_1 = 90 ; periodB_2 = 110

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
            "TS_JJA": (["time", "lat", "lon"], data),
        },
        coords={
            "time": (["time"], np.linspace(1850, 1850+149, 150)),
            "lat":  (["lat"],  src_file['lat'].data),
            "lon":  (["lon"],  src_file['lon'].data),
        },
        )

        ncfile['TS_JJA'].attrs = src_file['TS'].attrs
        ncfile.attrs['description']  =  'Created on 2023-10-16. This file is the JJA ensemble mean among the 8 member in the BTALnEU emission experiments. The variables are TS.'
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTALnEU_sulfate_column_burden_jja_mean_231017.nc", format='NETCDF4')

    out_path = '/exports/csce/datastore/geos/users/s2618078/data/model_data/TS/BTALnEU/'
    member_num = 8
    def cdo_ensemble(path, exp_name, var_name, member_num):
        '''This function cdo the emsemble member result'''
        import os

        for i in range(member_num):
            path_src = path + exp_name + '_' + str(i + 1) + '/mon/atm/' + var_name + '/'
            
            os.system('cdo cat ' + path_src + '*nc ' + cal_function.out_path + 'BTALnEU_TS_1850_150years_member_' + str(i + 1) + '.nc')
    
    def cal_ensemble(var_name):
        '''This function calculate the ensemble mean among all the members'''
        file_list = os.listdir(cal_function.out_path)
        ref_file  = xr.open_dataset(cal_function.out_path + file_list[0])

        # === Claim the array for result ===
        so4 = np.zeros((1883, 96, 144))

        for i in range(cal_function.member_num):
            f0 = xr.open_dataset(cal_function.out_path + 'BTALnEU_TS_1850_150years_member_' + str(i + 1) +'.nc')

            so4 += f0['TS'].data/cal_function.member_num

        print('Successfully calculated result')

        # Write to nc file
        ncfile  =  xr.Dataset(
            {
                "TS": (["time", "lat", "lon"], so4),
            },
            coords={
                "time": (["time"], ref_file['time'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
            },
            )

        ncfile['TS'].attrs = ref_file['TS'].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-18. This file is the ensemble mean among the 8 member in the BTALnEU emission experiments. The variable is surface temperature.'
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTALnEU_TS_ensemble_mean_231018.nc", format='NETCDF4')

    file_name  = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTALnEU_TS_ensemble_mean_231018.nc'
    def cal_jja_mean():
        '''This function calculate JJA mean for the data, which has been cdo cat and Ensemble_Mean'''
        file0 = xr.open_dataset(cal_function.file_name)

        # === Claim 150 year array ===
        jja_mean = np.zeros((150, 96, 144))

        for yyyy in range(150):
            jja_mean[yyyy] = np.average(file0['TS'].data[yyyy * 12 + 5 : yyyy * 12 + 8], axis=0)
        
        print('JJA mean calculation succeed!')

        # === Write to ncfile ===
        ncfile  =  xr.Dataset(
            {
                "TS_JJA": (["time", "lat", "lon"], jja_mean),
            },
            coords={
                "time": (["time"], np.linspace(1850, 1999, 150)),
                "lat":  (["lat"],  file0['lat'].data),
                "lon":  (["lon"],  file0['lon'].data),
            },
            )

        ncfile['TS_JJA'].attrs = file0['TS'].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-18. This file is JJA mean for the ensemble-mean sulfate column burden. The variable is surface temperature.'
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTALnEU_TS_ensemble_mean_JJA_231018.nc", format='NETCDF4')

    def cal_two_periods_difference(data):
        avg_periodA = np.average(data[period.periodA_1 : period.periodA_2], axis=0)
        avg_periodB = np.average(data[period.periodB_1 : period.periodB_2], axis=0)

        return avg_periodB - avg_periodA

class plot_function:
    '''This class save the settings and function for painting'''
    # ======== Set Extent ==========
    #lonmin,lonmax,latmin,latmax  =  30,110,0,60
    lonmin,lonmax,latmin,latmax  =  40,115,-10,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # ======== Set Figure ==========
    proj       =  ccrs.PlateCarree()

#    viridis = cm.get_cmap('coolwarm', 11)
#    newcolors = viridis(np.linspace(0, 1, 11))
#    newcmp = ListedColormap(newcolors)
#    newcmp.set_under('white')

    def paint_jja_diff(data):
        '''This function paint the Diff aerosol JJA'''
        proj       =  ccrs.PlateCarree()
        ax         =  plt.subplot(projection=proj)

        # Tick setting
        cyclic_data, cyclic_lon = add_cyclic_point(data, coord=src_file['lon'].data)
        set_cartopy_tick(ax=ax,extent=plot_function.extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=15)

        im2  =  ax.contourf(cyclic_lon, src_file['lat'].data, cyclic_data, np.linspace(-0.6,0.6,13), cmap='coolwarm', alpha=1, extend='both')

        ax.coastlines(resolution='50m', lw=1.2)

        bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none', alpha=0.7)
        ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

        ax.set_ylabel("No_EU", fontsize=15)

        ax.set_title('1901-1921 to 1941-1961', fontsize=15)

        # Add colorbar
        plt.colorbar(im2, orientation='horizontal')

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/TS/BTALnEU_diff_JJA_Indian_TS.pdf')

def main():
    # === 1. CDO emsemble members files === !!Finished 2023-10-18-09:46!!
    #cal_function.cdo_ensemble('/exports/csce/datastore/geos/groups/aerosol/sabineCESM/BTALnEU/', 'BTALnEU', 'TS', 8)

    # === 2. Calculate ensemble mean among the members output === !! Finished 2023-10-18-09:52!!
    #cal_function.cal_ensemble('TS')

    # === 3. Calculate JJA mean === !!Finished 2023-10-18-09:56!! /exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTALnEU_TS_ensemble_mean_JJA_231018.nc
    #cal_function.cal_jja_mean()

    # === 4. Calculate difference between two given period === !! Finished 2023-10-18-10:02!!
#    file0 = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTALnEU_TS_ensemble_mean_JJA_231018.nc')
#    jja_mean_diff = cal_function.cal_two_periods_difference(file0['TS_JJA'].data)
#    print('Successfully calculated JJA difference')
#    print(np.nanmin(jja_mean_diff))
#    print(np.nanmax(jja_mean_diff))
#
#    # === 5. Write calculation result to ncfile ===
#    ncfile  =  xr.Dataset(
#        {
#            "TS_JJA_DIFF": (["lat", "lon"], jja_mean_diff),
#        },
#        coords={
#            "lat":  (["lat"],  src_file['lat'].data),
#            "lon":  (["lon"],  src_file['lon'].data),
#        },
#        )
###
#    ncfile['TS_JJA_DIFF'].attrs = src_file['TS'].attrs
#    ncfile.attrs['description']  =  'Created on 2023-10-18. This file is the difference for period (1900-1920) and (1940-1960) JJA ensemble mean among the 8 member in the BTALnEU emission experiments. The variables are TS_JJA.'
#    ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/BTALnEU_surface_temperature_jja_mean_diff_1900_1960_231018.nc", format='NETCDF4')
#    ncfile.attrs['script']       =  'paint_BTALnEU_change_Indian_surface_temperature_between_period_231017.py'
   # === 5 Step Succeed ===

    # === 6. painting ===
    file1 = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTALnEU_surface_temperature_jja_mean_diff_1900_1960_231018.nc')
    plot_function.paint_jja_diff(file1['TS_JJA_DIFF'].data)

if __name__ == '__main__':
    main()