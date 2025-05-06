'''
2023-11-1
This script is to cope with the NCEP precipitation data
Prect data is from 1891 to 2020
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

'''the two periods for compare '''
periodA_1 = 50 ; periodA_2 = 70
periodB_1 = 90 ; periodB_2 = 110
    

class calculation:
    '''The area range for calculation'''
    lonmin,lonmax,latmin,latmax  =  70,90,0,30

    def calculate_JJA_mean(file_path, file_name):
        
        f0       = xr.open_dataset(file_path + file_name)
        # 1. Claim the mean array
        shape    = f0['precip'].data.shape
        JJA_mean = np.zeros((129, shape[1], shape[2]))

        # 2. Calculate JJA mean for each year
        start_mon = 5 # June
        num_mon   = 3
        for yyyy in range(129):
            JJA_mean[yyyy] = np.average(f0['precip'].data[yyyy * 12 + start_mon : yyyy * 12 + start_mon + num_mon], axis=0)

        # 3. Write to nc Dataset
        ncfile  =  xr.Dataset(
            {
                "prect_JJA": (["time", "lat", "lon"], JJA_mean),
            },
            coords={
                "time": (["time"], np.linspace(1891, 1891 + 129, 129)),
                "lat":  (["lat"],  f0['lat'].data),
                "lon":  (["lon"],  f0['lon'].data),
            },
            )

        ncfile['prect_JJA'].attrs = f0['precip'].attrs
        #ncfile.attrs['description']  =  'Created on 2023-10-19. This file is JJA mean for the {}. Experiment is {}.'.format(var_name, exp_name)
        #ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/{}_{}_ensemble_mean_JJA_231019.nc".format(var_name, exp_name), format='NETCDF4')
        #ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_wind_231018.py'
        return ncfile

    def cal_two_periods_difference(data,):
        avg_periodA = np.average(data[periodA_1 : periodA_2], axis=0)
        avg_periodB = np.average(data[periodB_1 : periodB_2], axis=0)

        return avg_periodB - avg_periodA

    #def filter(data, n)
    def cal_area_mean(ncdata):
        '''This function calculate the area-average value, inputdata must be xarray Dataset'''
        # For the obervation
        # ncdata_slice = ncdata.sel(lon=slice(calculation.lonmin, calculation.lonmax), lat=slice(calculation.latmax, calculation.latmin))
        # For the model
        ncdata_slice = ncdata.sel(lon=slice(calculation.lonmin, calculation.lonmax), lat=slice(calculation.latmin, calculation.latmax))

        area_avg     = np.zeros((len(ncdata.time.data)))

        for yyyy in range(len(ncdata.time.data)):
            area_avg[yyyy] = np.nanmean(ncdata_slice['PRECT_JJA'].data[yyyy] * 86400000)

        return area_avg

    def cal_moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

class paint:
    def paint_time_series(data_0, data_1, w):
        '''
            This function plot the time-series for the 1-D data
            data0: Raw data
            data1: moving-average data
            w    : moving parameter
        '''
        fig, ax = plt.subplots()

        ax.plot(np.linspace(1891, 1891 + 129, 129), data_0 / 31 - np.average(data_0 / 31))
        length  = int((w-1) / 2)
        ax.plot(np.linspace(1891 + length, 1891 + 129 - length, 129 - length * 2, dtype=int), data_1 / 31 - np.average(data_1 / 31), color='red')

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/prect/GPCC/' + 'long_term_series.pdf')

    def paint_time_series_model(data_0, data_1, data_2, data_3, w):
        '''
            This function plot the time-series for the 1-D data
            data0: Raw data
            data1: moving-average data
            w    : moving parameter
        '''
        fig, ax = plt.subplots()

        ax.plot(np.linspace(1891, 1891 + 109, 109), data_0  - np.average(data_0), color='gray', linestyle='solid')
        ax.plot(np.linspace(1891, 1891 + 109, 109), data_2  - np.average(data_2), color='gray', linestyle='--')
        length  = int((w-1) / 2)
        ax.plot(np.linspace(1891 + length, 1891 + 109 - length, 109 - length * 2, dtype=int), data_1  - np.average(data_1 ), color='black', linewidth=2)
        ax.plot(np.linspace(1891 + length, 1891 + 109 - length, 109 - length * 2, dtype=int), data_3  - np.average(data_3 ), color='red', linewidth=2)

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/prect/GPCC/' + 'model_nEU_long_term_series.pdf')




def main():
    # ========== GPCC observation =======================
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ========== 1. Calculate JJA mean ==================
#    file_path = '/exports/csce/datastore/geos/users/s2618078/data/download/GPCC_NCEP_prect/'
#    file_name = 'precip.mon.total.1x1.v2020.nc'
#
#    JJA_prect = calculation.calculate_JJA_mean(file_path, file_name)
#
#    # ========== 2. Calculate Indian area mean ==========
#    JJA_prect_area = calculation.cal_area_mean(JJA_prect)
#    #print(JJA_prect_area)
#    # Moving parameter
#    w  =  11
#    JJA_prect_area_moving = calculation.cal_moving_average(JJA_prect_area, w) # Return 129 - 12 = 117 values
#    print(len(JJA_prect_area_moving))
#
#    # ========== 3. Plot the long-term series ===========
#    paint.paint_time_series(JJA_prect_area, JJA_prect_area_moving, w)

    # ========== Model Control Experiment =======================
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ========== 1. Read JJA mean datafile==================
    file_path1  =  '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    file_name1  =  'BTAL_precipitation_jja_mean_231005.nc'

    con_JJA_prect  =  xr.open_dataset(file_path1 + file_name1).sel(time=slice(1891, 1999)) #109 time series

    # ========== 2. Calculate Indian area mean =============
    JJA_prect_area_con = calculation.cal_area_mean(con_JJA_prect)
    #print(len(con_JJA_prect.time.data))

    # ========== 3. Calculate moving mean ==================
    w  =  11
    JJA_prect_area_con_moving = calculation.cal_moving_average(JJA_prect_area_con, w)

#    print(JJA_prect_area_con_moving)
#    # ========== 4. Plot the time series ===================
#    paint.paint_time_series_model(JJA_prect_area_con, JJA_prect_area_con_moving, w)

    # ========== Model Control Experiment =======================
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # ========== 1. Read JJA mean datafile==================
    file_path1  =  '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    file_name1  =  'noEU_precipitation_jja_mean_231005.nc'

    eu_JJA_prect  =  xr.open_dataset(file_path1 + file_name1).sel(time=slice(1891, 1999)) #109 time series

    # ========== 2. Calculate Indian area mean =============
    JJA_prect_area_eu = calculation.cal_area_mean(eu_JJA_prect)
    #print(len(con_JJA_prect.time.data))

    # ========== 3. Calculate moving mean ==================
    w  =  11
    JJA_prect_area_eu_moving = calculation.cal_moving_average(JJA_prect_area_eu, w)

    #print(JJA_prect_area_con_moving)
    # ========== 4. Plot the time series ===================
    paint.paint_time_series_model(JJA_prect_area_con, JJA_prect_area_con_moving, JJA_prect_area_eu, JJA_prect_area_eu_moving, w)




if __name__ == '__main__':
    main()