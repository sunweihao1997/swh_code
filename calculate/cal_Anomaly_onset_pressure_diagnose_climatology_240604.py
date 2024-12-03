'''
2024-6-4
This script is used to calculate the climatological pressure variable, given the month, years range and variable name 
'''
import os
import numpy as np
import xarray as xr
import sys

sys.path.append("/home/sun/mycode_copy/calculate")
from cal_Anomaly_onset_precipitation_evolution_240602 import screen_early_late

#  ================= File Information =====================
path_in   =  '/home/sun/mydown/ERA5/monthly_pressure/'

file_list =  os.listdir(path_in)

onset_day_file    = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years.nc")
onset_day_file_42 = onset_day_file.sel(year=slice(1980, 2021)) #42 years

ref_file          = xr.open_dataset(path_in + "ERA5_2000_monthly_pressure_UTVWZSH.nc")

# ==========================================================

# Function for calculating the average for the given time-range
def cal_climatology_pressure(varname, yearlist):
    # 1. Claim the array for the result
    lev = ref_file.level.data
    lat = ref_file.latitude.data
    lon = ref_file.longitude.data
    climate_array  =  np.zeros((12, len(lev), len(lat), len(lon))) # 12-months

    # 2. Read 1-yr file and add to the average array
    num_year = len(yearlist)

    for yy in yearlist:
        yy_str  = str(int(yy))

        filename= "ERA5_replace_monthly_pressure_UTVWZSH.nc".replace("replace", yy_str)

        f_1yr   = xr.open_dataset(path_in + filename)

        climate_array += (f_1yr[varname].data / num_year)

    # 3. Finish and return the result
    return climate_array

if __name__ == '__main__':
    # 1. Deal with the 42 years climate mean
    year_list_all   = np.linspace(1980, 2021, 2021-1980+1)
    early_years, late_years = screen_early_late(onset_day_file_42['onset_day'].data)

    #print(early_years)
    # 2. Define the variable name
    varname = ['z', 'q', 't', 'u', 'v', 'w']

    # 3. Send to calculation
    # 3.1 Claim the ncfile
    ncfile = xr.Dataset()
    for vvvv in varname:
        climate_avg   = cal_climatology_pressure(vvvv, year_list_all)
        climate_early = cal_climatology_pressure(vvvv, early_years)
        climate_late  = cal_climatology_pressure(vvvv, late_years)

        # 3.2 Save to ncfile Dataset
        ncfile[vvvv+"_climate"] = xr.DataArray(
            climate_avg,
            coords={
                "time":ref_file.time,
                "latitude":ref_file.latitude,
                "longitude":ref_file.longitude,
                "level":ref_file.level,
            },
            dims=["time", "level", "latitude", "longitude"],
        )

        ncfile[vvvv+"_early"] = xr.DataArray(
            climate_early,
            coords={
                "time":ref_file.time,
                "latitude":ref_file.latitude,
                "longitude":ref_file.longitude,
                "level":ref_file.level,
            },
            dims=["time", "level", "latitude", "longitude"],
        )

        ncfile[vvvv+"_late"] = xr.DataArray(
            climate_late,
            coords={
                "time":ref_file.time,
                "latitude":ref_file.latitude,
                "longitude":ref_file.longitude,
                "level":ref_file.level,
            },
            dims=["time", "level", "latitude", "longitude"],
        )

        # 3.3 copy the attributes
        ncfile[vvvv+"_climate"].attrs = ref_file[vvvv].attrs
        ncfile[vvvv+"_early"].attrs   = ref_file[vvvv].attrs
        ncfile[vvvv+"_late"].attrs    = ref_file[vvvv].attrs

        print(f'Successfully calculated the {vvvv}')

    # 4. Write to the netcdf file
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_monsoon_onset_climate_early_late_pressure_variables_composite_average_monthly.nc")