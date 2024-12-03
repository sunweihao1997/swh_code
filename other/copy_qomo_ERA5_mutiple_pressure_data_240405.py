'''
2024-4-5
Because I need to use the Feb-Apr monthly pressure data, this script is to copy some variable data to my path
'''
import os

varname = ['divergence', 'temperature', 'geopotential', 'u_component_of_wind', 'v_component_of_wind']
year    = np.linspace(1980, 2020, 41, dtype=int)

for vv in varname:
    for yy in year:
        os.system('cp /data1/other_data/DataUpdate/ERA5/new-era5/monthly/multiple/' + str(yy) + '/' + vv + '_' + str(yy) + '01.nc /data4/2019swh/ERA5_pressure_Feb_Apr/')
        os.system('cp /data1/other_data/DataUpdate/ERA5/new-era5/monthly/multiple/' + str(yy) + '/' + vv + '_' + str(yy) + '02.nc /data4/2019swh/ERA5_pressure_Feb_Apr/')
        os.system('cp /data1/other_data/DataUpdate/ERA5/new-era5/monthly/multiple/' + str(yy) + '/' + vv + '_' + str(yy) + '03.nc /data4/2019swh/ERA5_pressure_Feb_Apr/')
        os.system('cp /data1/other_data/DataUpdate/ERA5/new-era5/monthly/multiple/' + str(yy) + '/' + vv + '_' + str(yy) + '04.nc /data4/2019swh/ERA5_pressure_Feb_Apr/')
        os.system('cp /data1/other_data/DataUpdate/ERA5/new-era5/monthly/multiple/' + str(yy) + '/' + vv + '_' + str(yy) + '05.nc /data4/2019swh/ERA5_pressure_Feb_Apr/')
        