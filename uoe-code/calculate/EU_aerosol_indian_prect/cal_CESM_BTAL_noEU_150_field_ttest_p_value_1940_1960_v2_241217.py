'''
2023-12-3
This script is to calculate ttest between two experiments for the period 1940 to 1960

the variable correspond to the variables in script: cal_CESM_BTAL_200_field_change_1900_1960_20231129.py

2023-12-4 modified:
Add calculating divergence to the generated ncfile
'''
import xarray as xr
import numpy as np
import sys

sys.path.append('/home/sun/uoe-code/module/')
from module_sun import cal_xydistance

# ========================== File location ================================

path_src = '/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/'

file_z_BTAL   = 'CESM_BTAL_JJA_Z3_ensemble.nc'
file_v_BTAL   = 'CESM_BTAL_JJA_V_ensemble.nc'
file_w_BTAL   = 'CESM_BTAL_JJA_OMEGA_ensemble.nc'
file_u_BTAL   = 'CESM_BTAL_JJA_U_ensemble.nc'
file_t_BTAL   = 'CESM_BTAL_JJA_T_ensemble.nc'
file_r_BTAL   = 'CESM_BTAL_JJA_RH_ensemble.nc'

file_z_BTALnEU   = 'CESM_BTALnEU_JJA_Z3_ensemble.nc'
file_v_BTALnEU   = 'CESM_BTALnEU_JJA_V_ensemble.nc'
file_w_BTALnEU   = 'CESM_BTALnEU_JJA_OMEGA_ensemble.nc'
file_u_BTALnEU   = 'CESM_BTALnEU_JJA_U_ensemble.nc'
file_t_BTALnEU   = 'CESM_BTALnEU_JJA_T_ensemble.nc'
file_r_BTALnEU   = 'CESM_BTALnEU_JJA_RH_ensemble.nc'

# =========================================================================

# ========================== File information =============================

ref_file = xr.open_dataset(path_src + 'CESM_BTAL_JJA_OMEGA_ensemble.nc')
lat      = ref_file.lat.data
lon      = ref_file.lon.data
time     = ref_file.time.data


# ========================== Calculation ==================================

def calculate_ensemble_average(ncfile, varname):
    '''
        This function calculate ensemble average
    '''

    # 1. Claim the array, notice this is only single layer value
    avg = np.zeros((len(time), len(lat), len(lon)))

    for i in range(8):
        avg += ncfile["JJA_" + varname + "_{}".format(i+1)].data / 8

    return avg

def calculate_ttest(ncfile1_all, ncfile1_sel, ncfile2_all, ncfile2_sel, varname):
    '''
        This function deal with golbal data with 8 ensemble members
    '''
    from scipy import stats
    # Claim the p-value array
    p_value = np.zeros((len(lat), len(lon)))

    for yy in range(len(lat)):
        for xx in range(len(lon)):
            anomaly_1 = ncfile1_sel["btal_" + varname].data[:, yy, xx] - np.average(ncfile1_all["btal_" + varname].data[:, yy, xx], axis=0)
            anomaly_2 = ncfile2_sel["btalneu_" + varname].data[:, yy, xx] - np.average(ncfile2_all["btalneu_" + varname].data[:, yy, xx], axis=0)

            a,b  = stats.ttest_ind(anomaly_1, anomaly_2, equal_var=False)
            p_value[yy, xx] = b
    
    return p_value

# ========================= Main function ================================

def calculation():

    f_z_BTAL = xr.open_dataset(path_src + file_z_BTAL).sel(lev=150)
    f_v_BTAL = xr.open_dataset(path_src + file_v_BTAL).sel(lev=150)
    f_w_BTAL = xr.open_dataset(path_src + file_w_BTAL).sel(lev=500)
    f_u_BTAL = xr.open_dataset(path_src + file_u_BTAL).sel(lev=150)
    f_z_BTALnEU = xr.open_dataset(path_src + file_z_BTALnEU).sel(lev=150)
    f_v_BTALnEU = xr.open_dataset(path_src + file_v_BTALnEU).sel(lev=150)
    f_w_BTALnEU = xr.open_dataset(path_src + file_w_BTALnEU).sel(lev=500)
    f_u_BTALnEU = xr.open_dataset(path_src + file_u_BTALnEU).sel(lev=150)


    # calculate ensemble average
    #print(f_z_BTAL)
    btal_z = calculate_ensemble_average(f_z_BTAL, "Z3")
    btal_v = calculate_ensemble_average(f_v_BTAL, "V")
    btal_w = calculate_ensemble_average(f_w_BTAL, "OMEGA")
    btal_u = calculate_ensemble_average(f_u_BTAL, "U")
    btalneu_z = calculate_ensemble_average(f_z_BTALnEU, "Z3")
    btalneu_v = calculate_ensemble_average(f_v_BTALnEU, "V")
    btalneu_w = calculate_ensemble_average(f_w_BTALnEU, "OMEGA")
    btalneu_u = calculate_ensemble_average(f_u_BTALnEU, "U")

    # save them to the xarray Datasets
    ncfile  =  xr.Dataset(
        {
            "btal_z": (["time", "lat", "lon"], btal_z),
            "btal_v": (["time", "lat", "lon"], btal_v),
            "btal_w": (["time", "lat", "lon"], btal_w),
            "btal_u": (["time", "lat", "lon"], btal_u),
            "btalneu_z": (["time", "lat", "lon"], btalneu_z),
            "btalneu_w": (["time", "lat", "lon"], btalneu_w),
            "btalneu_u": (["time", "lat", "lon"], btalneu_u),
            "btalneu_v": (["time", "lat", "lon"], btalneu_v),
        },
        coords={
            "time": (["time"], time),
            "lat":  (["lat"],  lat),
            "lon":  (["lon"],  lon),
        },
        )

    # Here I will calculate divergence using u/v at this level
    # BTAL
    disy,disx,location = cal_xydistance(lat,lon)
    vy = np.gradient(ncfile['btal_v'].data, location, axis=1)
    print('dimension vy is {}'.format(vy.shape))
    ux = vy.copy()
    for i in range(1, len(lat) - 1):
        ux[:, i, :] = np.gradient(ncfile['btal_u'].data[:, i, :], disx[i], axis=1)

    ncfile["btal_div"] = xr.DataArray(data=(ux + vy), dims=["time", "lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat), time=(["time"], time)), attrs=dict(description="divergence at this level",),)
    
    disy,disx,location = cal_xydistance(lat,lon)
    vy = np.gradient(ncfile['btalneu_v'].data, location, axis=1)
    ux = vy.copy()
    for i in range(1, len(lat) - 1):
        ux[:, i, :] = np.gradient(ncfile['btalneu_u'].data[:, i, :], disx[i], axis=1)

    ncfile["btalneu_div"] = xr.DataArray(data=(ux + vy), dims=["time", "lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat), time=(["time"], time)), attrs=dict(description="divergence at this level",),)


    # Set period for calculating anomaly
    period1 = 1945 ; period2 = 1960

    ncfile2 = ncfile.sel(time=slice(period1, period2))

    p_z = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="z")
    p_v = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="v")
    p_w = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="w")
    p_u = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="u")
    p_div = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="div")

    ncfile['p_z'] = xr.DataArray(data=p_z, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_v'] = xr.DataArray(data=p_v, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_w'] = xr.DataArray(data=p_w, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_u'] = xr.DataArray(data=p_u, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_div'] = xr.DataArray(data=p_div, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)

    ncfile.attrs['description'] = 'UVZ and DIVERGENCEat 150 hPa, ttest for the period 1945-1960. Created by /home/sun/uoe-code/calculate/EU_aerosol_indian_prect/cal_CESM_BTAL_noEU_150_field_ttest_p_value_1940_1960_v2_241217.py.'
    # Save all of them to the netcdf
    out_path = '/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    ncfile.to_netcdf(out_path + 'EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_150_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960_JJA.nc')

if __name__ == '__main__':
    calculation()