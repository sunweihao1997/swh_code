'''
250318
This script is to calculate some statistical physical quantity from the diabatic heating rate
'''
import numpy as np
import xarray as xr
import os
import math
data_path = "/home/sun/mydown/ERA5/monthly_pressure_diabatic_heating/"

data_list = os.listdir(data_path) ; data_list.sort() #; print(data_list)
print(len(data_list))

# ========== Get the onset date file ============
onset_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")

def cal_quantity_func(f0):
    # 1. Screen out the Maritime Continent area
    Mari_extent = [15, -5, 105, 135]
    level_extent1= [100, 500] ; level_extent2 = [300, 500]
    f0_mari = f0.sel(latitude=slice(Mari_extent[0], Mari_extent[1]), longitude=slice(Mari_extent[2], Mari_extent[3]))

    level   = f0_mari['level'].data
    #print(level[5:17])

    # 2. Calculate Maritime average
    avg_diabatic = np.average(np.average(f0_mari['diabatic_heating'].data, axis=2), axis=2) # output shape (Time, Level)

    avg_diabatic0 = np.average(f0_mari.sel(level=slice(150, 400))['diabatic_heating'].data[2:4]) # Vertical average among 500-100 hPa

    # 3. Calculate the vertical gradient
    # 3.1 Time-average
    avg_diabatic_MA = np.average(avg_diabatic[2:4], axis=0)
    avg_diabatic_MA_grad = np.gradient(avg_diabatic_MA, level, axis=0)

    return avg_diabatic0, np.average(avg_diabatic_MA_grad[5:17])

# Claim the average array
avg_q1 = np.zeros((43)) ; avg_q1_grad = np.zeros((43))
j = 0
for ff in data_list:
    file0 = xr.open_dataset(data_path + ff)
    print(f"Now it is dealing with file {ff}")

    avg_q1[j], avg_q1_grad[j] = cal_quantity_func(file0)

    print(avg_q1[j])

    j += 1


ncfile  =  xr.Dataset(
{
    "q1_avg": (["year"], avg_q1),
    "avg_q1_grad": (["year"], avg_q1_grad),
},
coords={
    "year": (["year"], np.linspace(2022, 1980, 43)),
},
)
ncfile.attrs['description']  =  'Generated on 20250318 by /home/sun/swh_code/calculate/cal_Anomaly_Onset_diabatic_average_and_vertical_gradient_over_Maritime_250318.py. Vertical average is 300-500 average, while gradient is 100-500 gradient. All is for the Maritime continent.'
ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_Maritime_Continent_monthly_March_April_diabatic_heating_statistical_quantity.nc")