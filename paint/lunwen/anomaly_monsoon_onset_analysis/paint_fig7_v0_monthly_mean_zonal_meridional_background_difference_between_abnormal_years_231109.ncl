'''
2023-11-09
This script is to calculate and plot the background difference in zonal and meridional directions between monsoon onset abnormal years

This script serves fot the last picture which will be used in the article
'''
from windspharm.xarray import VectorWind
import numpy as np
import xarray as xr
from matplotlib import projections
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cartopy
import cmasher as cmr
import os
import concurrent.futures

sys.path.append("/home/sun/mycode/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import set_cartopy_tick,save_fig

# -------------- 1. Data Path -----------------------
path0      =  "/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/"
path1      =  "/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/archive/"

vars_name  =  ["geopotential", "u_component_of_wind", "v_component_of_wind", "vertical_velocity", "relative_humidity", "temperature"]
variable_name  =  ["z", "u", "v", "w", "r", "t"]

# ================== Calculation ============================
# -------------- 2. Pre-process for data --------------------
def pre_process_data(var_filename, varname, path_in, path_out, name_out):
    '''
    This function is to archive daily-mean in onset early (or late) years data into a single file
    '''

    # ---------- 1. File lists -----------------
    all_files = os.listdir(path_in + var_filename)

    early_files = [] ; late_files = []
    for ffff in all_files:
        if "early" in ffff:
            early_files.append(ffff)

            continue

        elif "late" in ffff:
            late_files.append(ffff)

            continue

        else:
            print('Here is an astray file, whose name is {}'.format(ffff))

    if len(early_files)==365 and len(late_files)==365:
        early_files.sort() ; late_files.sort()
    else:
        print("The lenth of the file list is not 365, the early list is {} and late list is {}".format(len(early_files), len(late_files))) 

    # ---------- 2. Claim the array to save the data ------------
    ref_file         =  xr.open_dataset(path_in + var_filename + '/' + early_files[0])
    shapeinfo        =  ref_file[varname].data.shape # (1, 27, 181, 360)

    early_whole_year =  np.zeros((365, shapeinfo[1], shapeinfo[2], shapeinfo[3]))
    late_whole_year  =  early_whole_year.copy()

    # ---------- 3. Fill the data into the array ----------------
    print("Now it is start with variable {}".format(var_filename))
    for dddd in range(365):
        f_e = xr.open_dataset(path_in + var_filename + '/' + early_files[dddd])
        f_l = xr.open_dataset(path_in + var_filename + '/' + late_files[dddd])

        early_whole_year[dddd] = f_e[varname].data[0]
        late_whole_year[dddd]  = f_l[varname].data[0]

    # ---------- 4. Write the data to the file ------------------
    ncfile  =  xr.Dataset(
            {
                "early_whole_year_{}".format(varname): (["time", "lev", "lat", "lon"], early_whole_year),
                "late_whole_year_{}".format(varname):  (["time", "lev", "lat", "lon"], late_whole_year),
            },
            coords={
                "time": (["time"], np.linspace(1, 365, 365, dtype=int)),
                "lev":  (["lev"],  ref_file['lev'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
            },
            )

    ncfile["early_whole_year_{}".format(varname)].attrs = ref_file[varname].attrs
    ncfile["late_whole_year_{}".format(varname)].attrs  = ref_file[varname].attrs

    ncfile.attrs['description'] = 'Src_path is {}, just fill the daily result into a whole years file'.format(path_in)
    ncfile.attrs['script']      = 'paint_fig7_v0_monthly_mean_zonal_meridional_background_difference_between_abnormal_years_231109.py'

    ncfile.to_netcdf(path_out + name_out)

# -------------- 3. Calculate monthly data ------------------
def calculate_monthly_data(file_path, file_name, var_name):
    '''
    This function calculate monthly data for the input data
    '''
    month_day = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    f0        = xr.open_dataset(file_path + file_name)
    shape     = f0[var_name].data.shape

    # 1. Claim the base array
    mon_avg   = np.zeros((12, shape[1], shape[2], shape[3]))

    # 2. calculation
    for i in range(12):
        #print('Now it is calculate average from {} to {}'.format(sum(month_day[:i]), sum(month_day[:i + 1])))
        mon_avg[i] = np.average(f0[var_name].data[sum(month_day[:i]) : sum(month_day[:i + 1])])

    return mon_avg

# ======================= Main function ==============================
def calculate():
    # 1. Process the data into one file

    #with concurrent.futures.ProcessPoolExecutor() as executor:
    #    executor.submit(pre_process_data, var_filename=vars_name[0], varname=variable_name[0], path_in=path0, path_out=path1, name_out=vars_name[0]+'.nc')
    #    executor.submit(pre_process_data, var_filename=vars_name[1], varname=variable_name[1], path_in=path0, path_out=path1, name_out=vars_name[1]+'.nc')
    #    executor.submit(pre_process_data, var_filename=vars_name[2], varname=variable_name[2], path_in=path0, path_out=path1, name_out=vars_name[2]+'.nc')
    #    executor.submit(pre_process_data, var_filename=vars_name[3], varname=variable_name[3], path_in=path0, path_out=path1, name_out=vars_name[3]+'.nc')
    #    executor.submit(pre_process_data, var_filename=vars_name[4], varname=variable_name[4], path_in=path0, path_out=path1, name_out=vars_name[4]+'.nc')
    #    executor.submit(pre_process_data, var_filename=vars_name[5], varname=variable_name[5], path_in=path0, path_out=path1, name_out=vars_name[5]+'.nc')

    # ========== Step 1 completed, file saved in /home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/archive =============

    # 2. Calculate monthly average
    src_path = "/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/archive/daily/"
    with concurrent.futures.ProcessPoolExecutor() as executor:
        a0 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[0] + '.nc', var_name="early_whole_year_" + variable_name[0])
        a1 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[1] + '.nc', var_name="early_whole_year_" + variable_name[1])
        a2 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[2] + '.nc', var_name="early_whole_year_" + variable_name[2])
        a3 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[3] + '.nc', var_name="early_whole_year_" + variable_name[3])
        a4 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[4] + '.nc', var_name="early_whole_year_" + variable_name[4])
        a5 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[5] + '.nc', var_name="early_whole_year_" + variable_name[5])
        a6 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[0] + '.nc', var_name="late_whole_year_"  + variable_name[0])
        a7 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[1] + '.nc', var_name="late_whole_year_"  + variable_name[1])
        a8 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[2] + '.nc', var_name="late_whole_year_"  + variable_name[2])
        a9 = executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[3] + '.nc', var_name="late_whole_year_"  + variable_name[3])
        a10= executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[4] + '.nc', var_name="late_whole_year_"  + variable_name[4])
        a11= executor.submit(calculate_monthly_data, file_path=src_path, file_name=vars_name[5] + '.nc', var_name="late_whole_year_"  + variable_name[5])
    
    # Write to the ncfile
    # ---------- 4. Write the data to the file ------------------
    ref_file =  xr.open_dataset(src_path + "v_component_of_wind.nc")
    ncfile   =  xr.Dataset(
            {
                "early_whole_year_z": (["time", "lev", "lat", "lon"], a0.result()),
                "late_whole_year_z":  (["time", "lev", "lat", "lon"], a6.result()),
                "early_whole_year_u": (["time", "lev", "lat", "lon"], a1.result()),
                "late_whole_year_u":  (["time", "lev", "lat", "lon"], a7.result()),
                "early_whole_year_v": (["time", "lev", "lat", "lon"], a2.result()),
                "late_whole_year_v":  (["time", "lev", "lat", "lon"], a8.result()),
                "early_whole_year_w": (["time", "lev", "lat", "lon"], a3.result()),
                "late_whole_year_w":  (["time", "lev", "lat", "lon"], a9.result()),
                "early_whole_year_r": (["time", "lev", "lat", "lon"], a4.result()),
                "late_whole_year_r":  (["time", "lev", "lat", "lon"], a10.result()),
                "early_whole_year_t": (["time", "lev", "lat", "lon"], a5.result()),
                "late_whole_year_t":  (["time", "lev", "lat", "lon"], a11.result()),
            },
            coords={
                "time": (["time"], np.linspace(1, 12, 12, dtype=int)),
                "lev":  (["lev"],  ref_file['lev'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
            },
            )

    ncfile.attrs['description'] = 'Src_path is {}, this is the monthly mean calculated from daily files'.format(src_path)
    ncfile.attrs['script']      = 'paint_fig7_v0_monthly_mean_zonal_meridional_background_difference_between_abnormal_years_231109.py'

    ncfile.to_netcdf("/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/archive/monthly/ERA5_abnormal_onset_years_monthly_average.nc")

def paint():
    # =========== 1. Prepare for the selected monthly mean ============
    select_month = slice(2, 4)

    # =========== 2. Read data ========================================
    f0  =  xr.open_dataset("/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/archive/monthly/ERA5_abnormal_onset_years_monthly_average.nc").sel(time=select_month)

    # =========== 3. Caclculate difference among the abnormal years ===
    diff_u  =  np.average(f0['early_whole_year_u'].data, axis=0) - np.average(f0['late_whole_year_u'].data, axis=0)
    diff_v  =  np.average(f0['early_whole_year_v'].data, axis=0) - np.average(f0['late_whole_year_v'].data, axis=0)
    diff_w  =  np.average(f0['early_whole_year_w'].data, axis=0) - np.average(f0['late_whole_year_w'].data, axis=0)

    # =========== 4. Write difference to nc ===========================
    src_path = "/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/archive/daily/"
    ref_file =  xr.open_dataset(src_path + "v_component_of_wind.nc")
    ncfile   =  xr.Dataset(
            {
                "diff_u":  (["lev", "lat", "lon"], diff_u),
                "diff_v":  (["lev", "lat", "lon"], diff_v),
                "diff_w":  (["lev", "lat", "lon"], diff_w),
            },
            coords={
                "lev":  (["lev"],  ref_file['lev'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
            },
            )

    ncfile.attrs['description'] = 'This file saves the difference in the background stream field among the month Feb-Apr'
    ncfile.attrs['script']      = 'paint_fig7_v0_monthly_mean_zonal_meridional_background_difference_between_abnormal_years_231109.py'

    ncfile.to_netcdf("/home/sun/data/ERA5_data_monsoon_onset/climatic_daily_ERA5_BOBSM_early_late_years/background_diff/ERA5_monthly_difference_variable.nc")


if __name__ == '__main__':
    #calculate()
    paint()