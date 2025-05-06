'''
2023-12-3
This script is to calculate ttest between two experiments for the period 1940 to 1960

the variable correspond to the variables in script: cal_CESM_BTAL_200_field_change_1900_1960_20231129.py

2023-12-4 modified:
Add variable at 850 hPa
'''
import xarray as xr
import numpy as np

# ========================== File location ================================

path_src = '/exports/csce/datastore/geos/users/s2618078/data/model_data/ensemble_JJAS/'

file_z_BTAL   = 'CESM_BTAL_JJAS_Z3_ensemble.nc'
file_v_BTAL   = 'CESM_BTAL_JJAS_V_ensemble.nc'
file_w_BTAL   = 'CESM_BTAL_JJAS_OMEGA_ensemble.nc'
file_u_BTAL   = 'CESM_BTAL_JJAS_U_ensemble.nc'
file_t_BTAL   = 'CESM_BTAL_JJAS_T_ensemble.nc'
file_r_BTAL   = 'CESM_BTAL_JJAS_RH_ensemble.nc'

file_z_BTALnEU   = 'CESM_BTALnEU_JJAS_Z3_ensemble.nc'
file_v_BTALnEU   = 'CESM_BTALnEU_JJAS_V_ensemble.nc'
file_w_BTALnEU   = 'CESM_BTALnEU_JJAS_OMEGA_ensemble.nc'
file_u_BTALnEU   = 'CESM_BTALnEU_JJAS_U_ensemble.nc'
file_t_BTALnEU   = 'CESM_BTALnEU_JJAS_T_ensemble.nc'
file_r_BTALnEU   = 'CESM_BTALnEU_JJAS_RH_ensemble.nc'

# =========================================================================

# ========================== File information =============================

ref_file = xr.open_dataset(path_src + 'CESM_BTAL_JJAS_OMEGA_ensemble.nc')
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
        avg += ncfile["JJAS_" + varname + "_{}".format(i+1)].data / 8

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

    level = 850
    f_z_BTAL = xr.open_dataset(path_src + file_z_BTAL).sel(lev=level)
    f_v_BTAL = xr.open_dataset(path_src + file_v_BTAL).sel(lev=level)
    f_w_BTAL = xr.open_dataset(path_src + file_w_BTAL).sel(lev=level)
    f_u_BTAL = xr.open_dataset(path_src + file_u_BTAL).sel(lev=level)
    f_t_BTAL = xr.open_dataset(path_src + file_t_BTAL).sel(lev=level)
    f_z_BTALnEU = xr.open_dataset(path_src + file_z_BTALnEU).sel(lev=level)
    f_v_BTALnEU = xr.open_dataset(path_src + file_v_BTALnEU).sel(lev=level)
    f_w_BTALnEU = xr.open_dataset(path_src + file_w_BTALnEU).sel(lev=level)
    f_u_BTALnEU = xr.open_dataset(path_src + file_u_BTALnEU).sel(lev=level)
    f_t_BTALnEU = xr.open_dataset(path_src + file_t_BTALnEU).sel(lev=level)

    print(f_z_BTAL)


    # calculate ensemble average
    #print(f_z_BTAL)
    btal_z = calculate_ensemble_average(f_z_BTAL, "Z3")
    btal_v = calculate_ensemble_average(f_v_BTAL, "V")
    btal_w = calculate_ensemble_average(f_w_BTAL, "OMEGA")
    btal_u = calculate_ensemble_average(f_u_BTAL, "U")
    btal_t = calculate_ensemble_average(f_t_BTAL, "T")
    btalneu_z = calculate_ensemble_average(f_z_BTALnEU, "Z3")
    btalneu_v = calculate_ensemble_average(f_v_BTALnEU, "V")
    btalneu_w = calculate_ensemble_average(f_w_BTALnEU, "OMEGA")
    btalneu_u = calculate_ensemble_average(f_u_BTALnEU, "U")
    btalneu_t = calculate_ensemble_average(f_t_BTALnEU, "T")

    # save them to the xarray Datasets
    ncfile  =  xr.Dataset(
        {
            "btal_z": (["time", "lat", "lon"], btal_z),
            "btal_v": (["time", "lat", "lon"], btal_v),
            "btal_w": (["time", "lat", "lon"], btal_w),
            "btal_u": (["time", "lat", "lon"], btal_u),
            "btal_t": (["time", "lat", "lon"], btal_t),
            "btalneu_z": (["time", "lat", "lon"], btalneu_z),
            "btalneu_w": (["time", "lat", "lon"], btalneu_w),
            "btalneu_u": (["time", "lat", "lon"], btalneu_u),
            "btalneu_v": (["time", "lat", "lon"], btalneu_v),
            "btalneu_t": (["time", "lat", "lon"], btalneu_t),
        },
        coords={
            "time": (["time"], time),
            "lat":  (["lat"],  lat),
            "lon":  (["lon"],  lon),
        },
        )

    # Set period for calculating anomaly
    period1 = 1945 ; period2 = 1960

    ncfile2 = ncfile.sel(time=slice(period1, period2))

    p_z = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="z")
    p_v = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="v")
    p_w = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="w")
    p_u = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="u")
    p_t = calculate_ttest(ncfile1_all=ncfile, ncfile1_sel=ncfile2, ncfile2_all=ncfile, ncfile2_sel=ncfile2, varname="t")

    ncfile['p_z'] = xr.DataArray(data=p_z, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_v'] = xr.DataArray(data=p_v, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_w'] = xr.DataArray(data=p_w, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_u'] = xr.DataArray(data=p_u, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)
    ncfile['p_t'] = xr.DataArray(data=p_t, dims=["lat", "lon"], coords=dict(lon=(["lon"], lon), lat=(["lat"], lat),), attrs=dict(description="1945-1960 for ensemble-mean",),)

    ncfile.attrs['description'] = 'UVZWT at 850 hPa, ttest for the period 1945-1960'
    # Save all of them to the netcdf
    out_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    ncfile.to_netcdf(out_path + 'EUI_CESM_BTAL_BTALnEU_850_level_but_single_level_ensemble_mean_UVWZT_and_ttest_1945_1960.nc')

if __name__ == '__main__':
    calculation()