'''
2023-10-23
This script calculate the Specific Humidity using model output T and RH
because the model output do not include the specific humidity variable

This script is to prepare the T and RH variable for calculating the SH variable
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

variables = ['OMEGA', 'T', 'Z3', 'U',]
exps      = ['BTAL', 'BTALnEU']

class calculate_class:
    out_path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/VAR/'
    member_num = 8

    def cdo_ensemble(path1, exp_name, var_name, member_num):
        '''This function cdo the emsemble member result'''
        import os

        out_path1 = calculate_class.out_path0.replace('VAR', var_name) + exp_name + '/' + var_name + '/'
        check_path(out_path1, 1)
        
        path0 = path1 + exp_name + '/' + var_name + '/' + exp_name + '_' + str(member_num + 1) + '/'
            
        os.system('cdo cat ' + path0 + '*nc ' + out_path1 + exp_name + '_' + var_name + '_' + '1850_150years_member_' + str(member_num + 1) + '.nc')

    ensemble_path0 = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/Specific_humidity/'
    def cal_ensemble(exp_name, var_name):
        '''This function calculate the ensemble mean among all the members'''
        ensemble_path1 = calculate_class.ensemble_path0 + exp_name + '/' # This path saves the ncfiles
        file_list = os.listdir(ensemble_path1)
        ref_file  = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_RELHUM_ensemble_mean_231019.nc')

        # === Claim the array for result ===
        var_3d = np.zeros((1883, 29, 96, 144))
        print('Hello')

        for i in range(calculate_class.member_num):
            f0 = xr.open_dataset(ensemble_path1 + exp_name + '_' + var_name + '_' + '1850_150years_member_' + str(i + 1) +'.nc')

            var_3d += f0[var_name].data/calculate_class.member_num

        print('Successfully calculated result for {} {}'.format(exp_name, var_name))

        # Write to nc file
        ncfile  =  xr.Dataset(
            {
                var_name: (["time", "lev", "lat", "lon"], var_3d),
            },
            coords={
                "time": (["time"], ref_file['time'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
                "lev":  (["lev"],  ref_file['lev'].data),
            },
            )

        #ncfile[var_name].attrs = ref_file[var_name].attrs
        ncfile[var_name].attrs['unit'] = 'g/kg'
        ncfile.attrs['description']  =  'Created on 2023-10-23. This file is the ensemble mean among the 8 member in the {} emission experiments. The variable is {}.'.format(exp_name, var_name)
        ncfile.attrs['script']       =  'cal_model_output_specific_humidity_231023.py'
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/{}_{}_ensemble_mean_231019.nc".format(exp_name, var_name), format='NETCDF4')

def calculate_moisture_transport(wind, q):
    return wind * q
    
def main():
    import time
    start = time.perf_counter()

    # =========== 1. calculate cdo cat different variables ================================
    src_path = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/'
    #for i in range(8):
    #    for vv in variables:
    #        for ee in exps:
    #            calculate_class.cdo_ensemble(path1=src_path, exp_name=ee, var_name=vv, member_num=i)

    # =========== 2. calculate ensemble mean ==============================================
    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    executor.submit(calculate_class.cal_ensemble, 'BTAL', 'Q')
    #    executor.submit(calculate_class.cal_ensemble, 'BTALnEU', 'Q')
    #calculate_class.cal_ensemble( 'BTAL', 'Q')
    # =========== Step 2 Finished on 2023/10/23/19:51 =====================================

    # =========== 3. Calculate ensemble mean moisture transportation using U/V and Q ======
    path0   =  "/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/"
    u_btal  =  xr.open_dataset(path0 + "BTAL_U_ensemble_mean_231019.nc")
    v_btal  =  xr.open_dataset(path0 + "BTAL_V_ensemble_mean_231019.nc")
    u_noeu  =  xr.open_dataset(path0 + "BTALnEU_U_ensemble_mean_231019.nc")
    v_noeu  =  xr.open_dataset(path0 + "BTALnEU_V_ensemble_mean_231019.nc")

    q_btal  =  xr.open_dataset(path0 + "BTAL_Q_ensemble_mean_231019.nc")
    q_noeu  =  xr.open_dataset(path0 + "BTALnEU_Q_ensemble_mean_231019.nc")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        s1  =  executor.submit(calculate_moisture_transport, u_btal['U'].data, q_btal['Q'].data)
        s2  =  executor.submit(calculate_moisture_transport, v_btal['V'].data, q_btal['Q'].data)
        s3  =  executor.submit(calculate_moisture_transport, u_noeu['U'].data, q_noeu['Q'].data)
        s4  =  executor.submit(calculate_moisture_transport, v_noeu['V'].data, q_noeu['Q'].data)
        ms_u_btal    =  s1.result()
        ms_v_btal    =  s2.result()
        ms_u_noeu    =  s3.result()
        ms_v_noeu    =  s4.result()

    # ============ 4. Write file to netcdf ================================================
    ncfile  =  xr.Dataset(
            {
                'transport_x_BTAL': (["time", "lev", "lat", "lon"], ms_u_btal/9.8),
                'transport_y_BTAL': (["time", "lev", "lat", "lon"], ms_v_btal/9.8),
                'transport_x_BTALnEU': (["time", "lev", "lat", "lon"], ms_u_noeu/9.8),
                'transport_y_BTALnEU': (["time", "lev", "lat", "lon"], ms_v_noeu/9.8),
            },
            coords={
                "time": (["time"], u_btal['time'].data),
                "lat":  (["lat"],  u_btal['lat'].data),
                "lon":  (["lon"],  u_btal['lon'].data),
                "lev":  (["lev"],  u_btal['lev'].data),
            },
            )

    #ncfile[var_name].attrs = ref_file[var_name].attrs
    ncfile['transport_x_BTAL'].attrs['unit'] = 'g m-1 s-1 hPa-1'
    ncfile['transport_y_BTAL'].attrs['unit'] = 'g m-1 s-1 hPa-1'
    ncfile['transport_x_BTALnEU'].attrs['unit'] = 'g m-1 s-1 hPa-1'
    ncfile['transport_y_BTALnEU'].attrs['unit'] = 'g m-1 s-1 hPa-1'
    ncfile.attrs['description']  =  'Created on 2023-10-24. Thhis file saves the moisture flux calculated using Specific Humidity and Wind'
    ncfile.attrs['script']       =  'cal_model_output_specific_humidity_231023.py'
    ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc", format='NETCDF4')

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} seconds')

if __name__ == '__main__':
    main()