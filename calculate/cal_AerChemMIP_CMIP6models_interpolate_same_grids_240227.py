'''
2024-2-27
This script is trying to Unify the resolutions of different climate models using interpolation.

2024-3-11 modified:
change target to the daily precipitation 

2024-3-13 modified:
Due to the large size of the EC-Earth, I change the time slice for only 1950-2014

'''
import os
import xarray as xr
import numpy as np

data_path    = '/home/sun/data/download_data/AerChemMIP/day_prect/cdocat/'

interp_path  = '/home/sun/data/download_data/AerChemMIP/day_prect/cdo_cat_samegrid_linear1.5/'

#models_label = ['EC-Earth3-AerChem', 'UKESM1-0-LL', 'GFDL-ESM4', 'MRI-ESM2', 'GISS-E2-1-G', 'CESM2-WACCM', 'BCC-ESM1', 'NorESM2-LM', 'MPI-ESM-1-2-HAM', 'MIROC6', 'CNRM-ESM']
models_label = ['EC-Earth3-AerChem', 'UKESM1-0-LL', 'GFDL-ESM4', 'MRI-ESM2', 'GISS-E2-1-G', 'NorESM2-LM', 'MPI-ESM-1-2-HAM', 'MIROC6', ]


def group_files_by_model(list_all, keyword):
    same_group = []

    for i in list_all:
        if keyword in i:
            same_group.append(i)
        else:
            continue
    
    same_group.sort()
    return same_group

def check_latitude(f0):
    print(f0.lat.data)

def check_longitude(f0):
    print(f0.lon.data)

def check_varname(f0):
    print(f0)

def unify_lat_lon(f0, new_lat, new_lon, filename):
    '''
        This function is to unify the lat/lon information for each inputed f0
    '''
    old_lat   = f0['lat'].data
    old_lon   = f0['lon'].data
    time_data = f0['time'].data

    f0_interp = f0.interp(lat = new_lat, lon=new_lon,)

    f0_interp.to_netcdf(interp_path + filename)


def main():
    # 1. Get all files:
    files_all = os.listdir(data_path)

    # 2. return the information about latitude and longitude
    # === Result: all of them -90 to 90 for lat, and 0 to 365 for lon. Varname is pr ===
#    for mm in models_label:
#        model_group = group_files_by_model(files_all, mm)
#
#        f_lat = xr.open_dataset(data_path + model_group[0])
#
#        print(f'The model {mm} latitude is : \n')
#        check_latitude(f_lat)

#    for mm in models_label:
#        model_group = group_files_by_model(files_all, mm)
#
#        f_lon = xr.open_dataset(data_path + model_group[0])
#
#        print(f'The model {mm} longitude is : \n')
#        check_longitude(f_lon)

#    for mm in models_label:
#        model_group = group_files_by_model(files_all, mm)
#
#        ff = xr.open_dataset(data_path + model_group[0])
#
#        print(f'The model {mm} longitude is : \n')
#        check_varname(ff)

    # 3. Interpolate
    # 1.5 x 1.5 resolution
    new_lat = np.linspace(-88.5, 88.5, 119)
    new_lon = np.linspace(0, 358.5, 240)

    complete_list = os.listdir(interp_path)

    for fff in files_all:
        if fff[0] == '.':
            continue
        elif fff in complete_list:
            continue
        else:
            ff0 = xr.open_dataset(data_path + fff)
            if 'historical' in fff:
                year_range = np.linspace(1980, 2014, 2014 - 1980 + 1)
                
                ff  = ff0.sel(time=ff0.time.dt.year.isin(year_range))
            else:
                year_range = np.linspace(2015, 2050, 2050 - 2015 + 1)

                
                ff  = ff0.sel(time=ff0.time.dt.year.isin(year_range))

            del ff0

            print(f'Now it is dealing with {fff}')

            unify_lat_lon(ff, new_lat, new_lon, fff)

            print(f'Successfully interpolate the file {fff}')

#    print(new_lat)
#    print(new_lon)


if __name__ == '__main__':
    main()