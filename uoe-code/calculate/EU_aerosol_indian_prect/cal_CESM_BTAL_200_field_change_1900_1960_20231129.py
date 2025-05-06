'''
2023-11-29
This script is to plot changes between 1900 and 1960, focusing on the 200 hPa

variables include:
1. Z3
2. v
3. OMEGA (optional)

Save the result of calculation to the file, only save the layer I want in order to skimp on storage
'''
import xarray as xr
import numpy as np

# ========================== File location ================================

path_src = '/exports/csce/datastore/geos/users/s2618078/data/model_data/ensemble_JJAS/'

file_z   = 'CESM_BTAL_JJAS_Z3_ensemble.nc'
file_v   = 'CESM_BTAL_JJAS_V_ensemble.nc'
file_w   = 'CESM_BTAL_JJAS_OMEGA_ensemble.nc'
file_u   = 'CESM_BTAL_JJAS_U_ensemble.nc'
file_t   = 'CESM_BTAL_JJAS_T_ensemble.nc'
file_r   = 'CESM_BTAL_JJAS_RH_ensemble.nc'

# =========================================================================

# ========================== File information =============================

ref_file = xr.open_dataset(path_src + 'CESM_BTAL_JJAS_OMEGA_ensemble.nc')
ref_lat  = ref_file.lat.data
ref_lon  = ref_file.lon.data

# ========================== Calculation ==================================

def cal_period_diff(ncfile, varname, period, level):
    '''
        period includes for entries: periodA and periodB
        level: which pressure level to be select
    '''

    ncfile_A = ncfile.sel(time=slice(period[0], period[1]), lev=level)
    ncfile_B = ncfile.sel(time=slice(period[2], period[3]), lev=level)

    return -1 * ((np.average(ncfile_A[varname], axis=0) - np.average(ncfile_B[varname], axis=0)))

def calculate_stipple(num, data):
    '''
        This function deal with golbal data with 8 ensemble members
    '''
    # Claim the p-value array
    p_value = np.zeros((len(ref_lat), len(ref_lon)))

    for yy in range(len(ref_lat)):
        for xx in range(len(ref_lon)):
            positive = 0 ; negative = 0
            for i in range(num):
                if data[i, yy, xx] > 0:
                    positive += 1
                else:
                    negative += 1

            if positive >= 6 or negative >= 6:
                p_value[yy, xx] = 1
    
    return p_value

# ========================= Main function ================================

def calculation():

    f_z = xr.open_dataset(path_src + file_z)
    f_v = xr.open_dataset(path_src + file_v)
    f_w = xr.open_dataset(path_src + file_w)
    f_u = xr.open_dataset(path_src + file_u)


    # Create an array to save the result
    # 1. Claim the array to save each member difference between these two periods
    diff_z = np.zeros((8, len(ref_lat), len(ref_lon)))
    diff_v = np.zeros((8, len(ref_lat), len(ref_lon)))
    diff_w = np.zeros((8, len(ref_lat), len(ref_lon)))
    diff_u = np.zeros((8, len(ref_lat), len(ref_lon)))
    diff_t = np.zeros((8, len(ref_lat), len(ref_lon)))
    diff_r = np.zeros((8, len(ref_lat), len(ref_lon)))

    num = 8
    for i in range(num):
        diff_z[i] = cal_period_diff(f_z, "JJAS_Z3_{}".format(i + 1),    [1900, 1920, 1940, 1960], 200)
        diff_v[i] = cal_period_diff(f_v, "JJAS_V_{}".format(i + 1),     [1900, 1920, 1940, 1960], 200)
        diff_w[i] = cal_period_diff(f_w, "JJAS_OMEGA_{}".format(i + 1), [1900, 1920, 1940, 1960], 500)
        diff_u[i] = cal_period_diff(f_u, "JJAS_U_{}".format(i + 1),     [1900, 1920, 1940, 1960], 200)

    print('The difference between two periods for three variables have finished!')

    stipple_z = calculate_stipple(num=8, data=diff_z)
    stipple_v = calculate_stipple(num=8, data=diff_v)
    stipple_w = calculate_stipple(num=8, data=diff_w)
    stipple_u = calculate_stipple(num=8, data=diff_u)

    print('The stippling calculation have finished!')


    ncfile  =  xr.Dataset(
        {
            "diff_z": (["member", "lat", "lon"], diff_z),
            "diff_v": (["member", "lat", "lon"], diff_v),
            "diff_u": (["member", "lat", "lon"], diff_u),
            "diff_w": (["member", "lat", "lon"], diff_w),
            "stipple_z": (["lat", "lon"], stipple_z),
            "stipple_v": (["lat", "lon"], stipple_v),
            "stipple_w": (["lat", "lon"], stipple_w),
            "stipple_u": (["lat", "lon"], stipple_u),
        },
        coords={
            "member": (["member"], np.linspace(1, 8, 8)),
            "lat":  (["lat"],  ref_lat),
            "lon":  (["lon"],  ref_lon),
        },
        )

    ncfile['diff_w'].attrs['description'] = 'Note: This is OMEGA at 500 hPa level!'

    ncfile.attrs['description'] = 'Created on 2023-11-29. This file saves the difference between two periods and stippling indicate that more than six member give the same signs for change. Please be cautious the OMEGA is 500 hPa.'
    ncfile.attrs['information'] = 'UOE-server: paint_CESM_BTAL_200_field_change_1900_1960_20231129.py'

    out_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'

    ncfile.to_netcdf(out_path + 'EUI_CESM_diff_BTAL_1900_1960_200hPa_Z_UV_500hPa_Omega.nc')

if __name__ == '__main__':
    calculation()