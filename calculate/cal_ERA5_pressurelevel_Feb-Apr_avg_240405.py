'''
2024-4-5
This script is to calculate the Feb-Apr averaged variables on pressure level
'''
import xarray as xr
import numpy as np

path_in = '/home/sun/data/download/part_ERA5_pressure_level/'

ref_file = xr.open_dataset(path_in + 'v_component_of_wind_201803.nc') ; interp_file = xr.open_dataset('/home/sun/data/download/ERA5_SLP_monthly/era5_monthly_single_mean_sea_level_pressure_2006.nc')
#print(ref_file.latitude.data) # 32, 721, 1440 !! Here I need to interpolate the file into 1x1 resolution

# Claim the array to save the result
array0 = np.zeros((41, 32, 181, 360))
array0 = np.zeros((41, 32, 721, 1440))

#varnames = ['divergence', 'u', 'v', 'geopotential', 'temperature']
#varss    = ['d', 'u', 'v', 'z', 't']
varnames = ['vertical_velocity']
varss    = ['w']

# --- Check the time line in the cdo cat file ---
f0       = xr.open_dataset(path_in + 'divergence_Jan-May_1980-2020.nc')
#print(f0.time.data) # !! Right !!

def cal_multiple_years_series(vvvv, varname):
    array1 = array0.copy() # The array saving the result

    f1 = xr.open_dataset(path_in + vvvv + '_Jan-May_1980-2020.nc')
    #f1 = f1.interp(longitude=interp_file.longitude.data, latitude=interp_file.latitude.data)

    y0 = 0
    for yyyy in range(1980, 2021):
        # 1. Extract one year data
        f1_1year = f1.sel(time=f1.time.dt.year.isin([yyyy]))

        #print(len(f1_1year.time.data)) # It is OK
        array1[y0] += np.average(f1_1year[varname].data[1:4], axis=0)

        y0 += 1

    return array1

#cal_multiple_years_series(varnames[1])
for num in range(len(varnames)):
    print(f'Now it is dealing with {varnames[num]}')
    a = cal_multiple_years_series(varnames[num], varss[num])

    # save it to the netCDF file and intepolate it
    ncfile  =  xr.Dataset(
        {
            varss[num]: (["year", "level", "latitude", "longitude"], a),
        },
    coords={
        "latitude": (["latitude"], ref_file.latitude.data),
        "longitude": (["longitude"], ref_file.longitude.data),
        "level": (["level"], ref_file.level.data),
        "year":  (["year"],  np.linspace(1980, 2020, 41)),
    },
    )
    ncfile.attrs['description']  =  'This file includes Feb-Mar-Apr average value for the 1980 to 2020. This data is used to calculate the regression between OLR/SLP index.'
    ncfile.attrs['script']  =  'cal_ERA5_pressurelevel_Feb-Apr_avg_240405.py'

    # Interpolate
    print(f'Now it is intepolate data')
    ncfile  =  ncfile.interp(longitude=interp_file['longitude'].data, latitude=interp_file['latitude'].data)

    ncfile.to_netcdf('/home/sun/data/download/part_ERA5_pressure_level/{}_FMA_1980-2020.nc'.format(varnames[num]))


    del a