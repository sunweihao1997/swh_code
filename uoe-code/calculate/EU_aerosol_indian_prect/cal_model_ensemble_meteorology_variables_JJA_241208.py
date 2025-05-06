'''
2023-11-27
This script is to calculate some variables from model experiments

experiment includes:
1. BTAL
2. Fix_EU

variables:
1. Z
2. UV
3. OMEGA
4. T
5. RELHUM
'''
import xarray as xr
import numpy as np
import os
import sys

# ================ file information ======================

path0 = '/home/sun/data/download_data/data/model_data/'
sub1  = ['OMEGA', 'RELHUM', 'T', 'Z3', 'WIND']
sub2  = ['BTAL', 'BTALnEU']
sub3  = ['U', 'V']

varname = ['OMEGA', 'RELHUM', 'T', 'Z3', 'U', 'V']
ref_file0 = xr.open_dataset('/home/sun/data/download_data/data/model_data/Z3/BTAL/Z3/BTAL_Z3_1850_150years_member_1.nc')

# ========================================================

# Function for calculation
def cal_season_average(path_src, firstsub, secondsub, thirdsub, member, varname):
    '''
        The input is ncfile
        member: Which ensemble, ranging from 1-8
    '''
    path = path_src + firstsub + '/' + secondsub + '/' + thirdsub + '/'
    file_list = os.listdir(path) ; file_list.sort()

    data = xr.open_dataset(path + file_list[member])

    # 1. Select the JJA data
    data_JJA = data.sel(time=data.time.dt.month.isin([7, 8, 9]))

    #print(data_JJA)
    # 2. Claim the array to save
    year_number = 157 ; shape = data_JJA[varname].data.shape

    if shape[0] != 157 * 3:
        print('ERROR! {} do not have correct number of year'.format(firstsub))
        sys.exit()

    JJA_prect = np.zeros((year_number, shape[1], shape[2], shape[3]))

    # 3. Calculation
    for yyyy in range(year_number):
        JJA_prect[yyyy] = np.average(data_JJA[varname].data[yyyy*3 : yyyy*3+3], axis=0)

    return JJA_prect, data_JJA[varname].attrs['units']

def save_to_ncfile(lists, varname, out_path, out_name, units):
    '''
        This function extract arrays from lists and then write them to the dataset
    '''
    dataset = xr.Dataset()

    for i in range(8):
        dataset[varname + str(i+1)] = xr.DataArray(
                    data=lists[i],
                    dims=["time", "lev", "lat", "lon"],
                    coords=dict(
                        lon=(["lon"], ref_file0.lon.data),
                        lat=(["lat"], ref_file0.lat.data),
                        lev=(["lev"], ref_file0.lev.data),
                        time=np.linspace(1850, 1850 + 156, 157),
                    ),
                    attrs=dict(
                        units=units,
                    ),
                )
    
    dataset.to_netcdf(out_path + out_name)

# =====================================BTAL======================================================
list0_omega = [] ; list1_omega = []
list0_z3    = [] ; list1_z3    = []
list0_u     = [] ; list1_u     = []
list0_v     = [] ; list1_v     = []
list0_rh    = [] ; list1_rh    = []
list0_t     = [] ; list1_t     = []

for i in range(8):
    print('Now it is dealing for member {}'.format(i + 1))
    # OMEGA
    a, b = cal_season_average(path_src=path0, firstsub=sub1[0], secondsub=sub2[0], thirdsub=sub1[0], member=i, varname=sub1[0])
    list0_omega.append(a) ; list1_omega.append(b)
    # RELHUM
    a, b = cal_season_average(path_src=path0, firstsub=sub1[1], secondsub=sub2[0], thirdsub=sub1[1], member=i, varname=sub1[1])
    list0_rh.append(a) ; list1_rh.append(b)
    # T
    a, b = cal_season_average(path_src=path0, firstsub=sub1[2], secondsub=sub2[0], thirdsub=sub1[2], member=i, varname=sub1[2])
    list0_t.append(a) ; list1_t.append(b)
    # Z3
    a, b = cal_season_average(path_src=path0, firstsub=sub1[3], secondsub=sub2[0], thirdsub=sub1[3], member=i, varname=sub1[3])
    list0_z3.append(a) ; list1_z3.append(b)
    # U
    a, b = cal_season_average(path_src=path0, firstsub=sub1[4], secondsub=sub2[0], thirdsub=sub3[0], member=i, varname=sub3[0])
    list0_u.append(a) ; list1_u.append(b)
    # V
    a, b = cal_season_average(path_src=path0, firstsub=sub1[4], secondsub=sub2[0], thirdsub=sub3[1], member=i, varname=sub3[1])
    list0_v.append(a) ; list1_v.append(b)

out0 = '/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/'
save_to_ncfile(lists=list0_z3, varname="JJA_Z3_", out_path=out0, out_name="CESM_BTAL_JJA_Z3_ensemble.nc", units=list1_z3[0])
save_to_ncfile(lists=list0_u,  varname="JJA_U_",  out_path=out0, out_name="CESM_BTAL_JJA_U_ensemble.nc",  units=list1_u[0])
save_to_ncfile(lists=list0_v,  varname="JJA_V_",  out_path=out0, out_name="CESM_BTAL_JJA_V_ensemble.nc",  units=list1_v[0])
save_to_ncfile(lists=list0_omega, varname="JJA_OMEGA_", out_path=out0, out_name="CESM_BTAL_JJA_OMEGA_ensemble.nc", units=list1_omega[0])
save_to_ncfile(lists=list0_rh, varname="JJA_RH_", out_path=out0,       out_name="CESM_BTAL_JJA_RH_ensemble.nc", units=list1_rh[0])
save_to_ncfile(lists=list0_t,  varname="JJA_T_",  out_path=out0,       out_name="CESM_BTAL_JJA_T_ensemble.nc",  units=list1_t[0])

# ============================================BTALnEU========================================================
list0_omega = [] ; list1_omega = []
list0_z3    = [] ; list1_z3    = []
list0_u     = [] ; list1_u     = []
list0_v     = [] ; list1_v     = []
list0_rh    = [] ; list1_rh    = []
list0_t     = [] ; list1_t     = []

for i in range(8):
    print('Now it is dealing for member {}'.format(i + 1))
    # OMEGA
    a, b = cal_season_average(path_src=path0, firstsub=sub1[0], secondsub=sub2[1], thirdsub=sub1[0], member=i, varname=sub1[0])
    list0_omega.append(a) ; list1_omega.append(b)
    # RELHUM
    a, b = cal_season_average(path_src=path0, firstsub=sub1[1], secondsub=sub2[1], thirdsub=sub1[1], member=i, varname=sub1[1])
    list0_rh.append(a) ; list1_rh.append(b)
    # T
    a, b = cal_season_average(path_src=path0, firstsub=sub1[2], secondsub=sub2[1], thirdsub=sub1[2], member=i, varname=sub1[2])
    list0_t.append(a) ; list1_t.append(b)
    # Z3
    a, b = cal_season_average(path_src=path0, firstsub=sub1[3], secondsub=sub2[1], thirdsub=sub1[3], member=i, varname=sub1[3])
    list0_z3.append(a) ; list1_z3.append(b)
    # U
    a, b = cal_season_average(path_src=path0, firstsub=sub1[4], secondsub=sub2[1], thirdsub=sub3[0], member=i, varname=sub3[0])
    list0_u.append(a) ; list1_u.append(b)
    # V
    a, b = cal_season_average(path_src=path0, firstsub=sub1[4], secondsub=sub2[1], thirdsub=sub3[1], member=i, varname=sub3[1])
    list0_v.append(a) ; list1_v.append(b)

out0 = '/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/'
save_to_ncfile(lists=list0_z3, varname="JJA_Z3_", out_path=out0, out_name="CESM_BTALnEU_JJA_Z3_ensemble.nc", units=list1_z3[0])
save_to_ncfile(lists=list0_u,  varname="JJA_U_",  out_path=out0, out_name="CESM_BTALnEU_JJA_U_ensemble.nc",  units=list1_u[0])
save_to_ncfile(lists=list0_v,  varname="JJA_V_",  out_path=out0, out_name="CESM_BTALnEU_JJA_V_ensemble.nc",  units=list1_v[0])
save_to_ncfile(lists=list0_omega, varname="JJA_OMEGA_", out_path=out0, out_name="CESM_BTALnEU_JJA_OMEGA_ensemble.nc", units=list1_omega[0])
save_to_ncfile(lists=list0_rh, varname="JJA_RH_", out_path=out0, out_name="CESM_BTALnEU_JJA_RH_ensemble.nc", units=list1_rh[0])
save_to_ncfile(lists=list0_t,  varname="JJA_T_",  out_path=out0, out_name="CESM_BTALnEU_JJA_T_ensemble.nc",  units=list1_t[0])