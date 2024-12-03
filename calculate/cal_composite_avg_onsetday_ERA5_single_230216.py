'''
2023-2-16
This script calculate the composite average for single layer using ERA5 data
'''
import xarray as xr
import numpy as np
import os
import sys
path0      =  '/data1/other_data/DataUpdate/ERA5/new-era5-single-daily-float-yearly/'

file_names = [
    '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature', 'sea_surface_temperature', 'skin_temperature', 'surface_latent_heat_flux', 'surface_net_solar_radiation', 'surface_pressure', 'surface_sensible_heat_flux',
    'total_precipitation'
]

var_names  = [
    'u10', 'v10', 't2m', 'sst', 'skt', 'slhf', 'ssr', 'sp', 'sshf', 'tp'
]

ref_file0  = xr.open_dataset(path0 + 'total_precipitation_2009.nc')
year_range = [1980, 2019]

end_path  =  '/data5/2019swh/data/composite_ERA5/single/'

# 1. Read the onset day data
dates = xr.open_dataset('/data5/2019swh/data/onsetdate.nc')

# calculate the early/late year number
early_years = []
late_years  = []
for i in range(1980, 2020):
    if dates['bob_onset_date'].data[i-1980] < np.average(dates['bob_onset_date'].data) - np.std(dates['bob_onset_date'].data):
	    early_years.append(i)
    elif dates['bob_onset_date'].data[i-1980] > np.average(dates['bob_onset_date'].data) + np.std(dates['bob_onset_date'].data):
	    late_years.append(i)

print(early_years)
print(late_years)
#sys.exit()
# 2. Calculate the composite average
def calculate_single_composite(filename, varname, year_range,):
    '''
        For the single layer data is saved for file of 365 time long,
        Here I define a base array and add each year from the d0-30 position
    '''
    ref_file0 = xr.open_dataset(path0 + filename + '_2009.nc')

    # 2.1 Claim the base array
    base_array = np.zeros((80, ref_file0[varname].shape[1], ref_file0[varname].shape[2]))

    # 2.2 Calculation
    divide_year = year_range[1] - year_range[0] + 1

    for yyyy in range(year_range[0], year_range[1] + 1):
        #print(int(dates.sel(year=yyyy)['bob_onset_date'].data))
        f1 = xr.open_dataset(path0 + filename + '_' + str(yyyy) + '.nc')
        base_array += f1[varname].data[int(dates.sel(year=yyyy)['bob_onset_date'].data) - 70 : int(dates.sel(year=yyyy)['bob_onset_date'].data) + 10] / divide_year

    # 2.3 Save to the ncfile
    ncfile  =  xr.Dataset(
                    {
                        varname: (["time", "lat", "lon"], base_array),
                    },
                    coords={
                        "lat": (["lat"], ref_file0.latitude.data),
                        "lon": (["lon"], ref_file0.longitude.data),
                        "time": (["time"], np.linspace(-70, 9, 80)),
                    },
                    )
    ncfile[varname].attrs = ref_file0[varname].attrs
    ncfile['lat'].attrs = ref_file0.latitude.attrs
    ncfile['lon'].attrs = ref_file0.longitude.attrs
    ncfile.attrs['description'] = 'created on 2024-8-16. This is composite data averaged from 1980 to 2019 using ERA5 data. Time range is -70 to 9'

    ncfile.to_netcdf(end_path + varname + '_composite.nc')

# 3. Calculate the composite average in abnormal years
def calculate_single_composite_abnormal(filename, varname, abnormal_years):
    '''
        For the single layer data is saved for file of 365 time long,
        Here I define a base array and add each year from the d0-30 position
    '''
    ref_file0 = xr.open_dataset(path0 + filename + '_2009.nc')

    # 2.1 Claim the base array
    base_array = np.zeros((80, ref_file0[varname].shape[1], ref_file0[varname].shape[2]))

    # 2.2 Calculation
    if 'early' in abnormal_years:
        divide_year = len(early_years)
        abnormal_yyyy = early_years
    elif 'late' in abnormal_years:
        divide_year = len(late_years)
        abnormal_yyyy = late_years


    for yyyy in abnormal_yyyy:
        #print(int(dates.sel(year=yyyy)['bob_onset_date'].data))
        f1 = xr.open_dataset(path0 + filename + '_' + str(yyyy) + '.nc')
        base_array += f1[varname].data[int(dates.sel(year=yyyy)['bob_onset_date'].data) - 70 : int(dates.sel(year=yyyy)['bob_onset_date'].data) + 10] / divide_year

    # 2.3 Save to the ncfile
    ncfile  =  xr.Dataset(
                    {
                        varname: (["time", "lat", "lon"], base_array),
                    },
                    coords={
                        "lat": (["lat"], ref_file0.latitude.data),
                        "lon": (["lon"], ref_file0.longitude.data),
                        "time": (["time"], np.linspace(-70, 9, 80)),
                    },
                    )
    ncfile[varname].attrs = ref_file0[varname].attrs
    ncfile['lat'].attrs = ref_file0.latitude.attrs
    ncfile['lon'].attrs = ref_file0.longitude.attrs
    ncfile.attrs['description'] = 'created on 2024-8-16. This is composite data averaged from 1980 to 2019 using ERA5 data.time range is -70 to 9.'

    ncfile.to_netcdf(end_path + varname + '_composite_'+ abnormal_years +'.nc')

def main():
    for nnnn in range(len(file_names)):
        calculate_single_composite(file_names[nnnn], var_names[nnnn], year_range=[1980, 2019],)
        calculate_single_composite_abnormal(file_names[nnnn], var_names[nnnn], abnormal_years='year_early',)
        calculate_single_composite_abnormal(file_names[nnnn], var_names[nnnn], abnormal_years='year_late',)

if __name__ == '__main__':
    main()
