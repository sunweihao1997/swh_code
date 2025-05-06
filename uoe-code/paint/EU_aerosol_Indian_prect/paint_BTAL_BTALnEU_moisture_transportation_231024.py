'''
2023-10-24
This script is gonna to plot difference between BTAL and BTALnEU in the variable moisture transportation and its convergence

The source file is /exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc

Outline:
1. calculate JJA mean
2. calculate difference between two period
3. calculate difference in trend between two experiments
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
import concurrent.futures

# select two period to calculate difference
class period:
    ''' This class infludes the two periods for compare '''
    periodA_1 = 50 ; periodA_2 = 70
    periodB_1 = 90 ; periodB_2 = 110

class calculate_class:
    def cal_jja_mean(file_path, file_name, var_name, exp_name,):
        '''This function calculate JJA mean for the data, which has been cdo cat and Ensemble_Mean'''
        file0 = xr.open_dataset(file_path + file_name)

        # === Claim 150 year array ===
        jja_mean = np.zeros((150, 29, 96, 144))

        for yyyy in range(150):
            jja_mean[yyyy] = np.average(file0[var_name].data[yyyy * 12 + 5 : yyyy * 12 + 8], axis=0)
        
        print('{} {} JJA mean calculation succeed!'.format(exp_name, var_name))

        # === Write to ncfile ===
        ncfile  =  xr.Dataset(
            {
                "{}_JJA".format(var_name): (["time", "lev", "lat", "lon"], jja_mean),
            },
            coords={
                "time": (["time"], np.linspace(1850, 1999, 150)),
                "lat":  (["lat"],  file0['lat'].data),
                "lon":  (["lon"],  file0['lon'].data),
                "lev":  (["lev"],  file0['lev'].data),
            },
            )

        ncfile["{}_JJA".format(var_name)].attrs = file0[var_name].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-19. This file is JJA mean for the {}. Experiment is {}.'.format(var_name, exp_name)
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/{}_{}_ensemble_mean_JJA_231019.nc".format(var_name, exp_name), format='NETCDF4')
        ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_wind_231018.py'

    def cal_two_periods_difference(data, dim):
        ref_file    = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/model_data/WIND/BTAL/U/BTAL_U_1850_150years_member_1.nc')

        avg_periodA = np.average(data[period.periodA_1 : period.periodA_2], axis=0)
        avg_periodB = np.average(data[period.periodB_1 : period.periodB_2], axis=0)

        # Add coordination to the array
        if dim==3:
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
        else:
            ncfile  =  xr.Dataset(
                {
                    "jja_diff": (["lat", "lon"], avg_periodB - avg_periodA),
                },
                coords={
                    "lat":  (["lat"],  ref_file['lat'].data),
                    "lon":  (["lon"],  ref_file['lon'].data),
                },
                )
        
        return ncfile

