'''
20230827
This script is to calculate horizontal correlation between onsetdate and LSTC index, time resolution is pentad.
Because it is hard to select a good area to maintain the good correlation
'''
import numpy as np
import xarray as xr
import os
from scipy import stats
from scipy import signal

# ============ First. Calculate the array for the LSTC index =======================
path0 = '/home/sun/data/other_data/single/pentad/slp/'

mask_file = xr.open_dataset('/home/sun/data/mask/ERA5_land_sea_mask_1x1.nc')
onset_date = xr.open_dataset('/home/sun/data/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc')
# ====== Claim the array ================================
LSTC_pentad = np.zeros((83, 73))

# ====== Divide the Land Sea Mask ======================
land_lon = slice(70, 90)
land_lat = slice(20, 5)

mask_land = mask_file.sel(longitude=land_lon, latitude=land_lat) # No ocean grid in the ocean area, do not need mask

file_path = '/home/sun/data/other_data/single/pentad/slp/'
file_list = os.listdir(file_path) ; file_list.sort()

def cal_LSTC_index(ffff):
    f_land = xr.open_dataset(file_path + ffff).sel(latitude=land_lat, longitude=land_lon)

    # Mask the ocean value in each pentad
    for tt in range(73):
        f_land['msl'].data[tt, ][mask_land['lsm'].data[0] < 0.8] = np.nan

    LSTC_index = np.zeros(73)

    for tt in range(73):
        LSTC_index[tt] = np.nanmean(f_land['msl'].data[tt])

    return LSTC_index

# ====== Calculation ==================================
LSTC = np.zeros((83, 73))
for yy in range(83):
    LSTC[yy] = cal_LSTC_index(file_list[yy])

# =========== Second. single point correlation calculation =================
correlation = np.zeros((73, 181, 360))
for tt in range(40):
    print('Now it is deal with {}'.format(tt))
    # 1. Firstly, I should make new array to get together all years in each pentad
    single_pentad_array = np.zeros((83, 181, 360))
    for yy in range(83):
        f0 = xr.open_dataset(file_path + file_list[yy])

        single_pentad_array[yy] = f0['msl'].data[tt]
    # 2. Secondly, calculate single point correlation in this pentad
    for ii in range(181):
        for jj in range(360):
            corre = stats.pearsonr(signal.detrend(onset_date['onset_day'].data) - np.average(signal.detrend(onset_date['onset_day'].data)), signal.detrend(single_pentad_array[:, ii, jj]) - np.average(signal.detrend(single_pentad_array[:, ii, jj])))
            correlation[tt, ii, jj] = corre[0]

# =========== Thirdly, write to the file ===================================
ncfile  =  xr.Dataset(
                    {
                        'correlation': (["time","lat","lon"], correlation),
                    },
                    coords={
                        "time": (["time"], np.linspace(1,73,73)),
                        "lat":  (["lat"],  f0.latitude.data),
                        "lon":  (["lon"],  f0.longitude.data),
                    },
                    )
ncfile.attrs['description'] = 'created on 2023-8-27. This file includes the correlation between onset date and LSTC index in each single per pentad. Time resolution is pentad 0-40. All the variable has been detrend'

ncfile.to_netcdf('/home/sun/data/ERA5_data_monsoon_onset/index/correlation/onset_dates_with_new_pentad_LSTC_detrend_horizontal_allyear.nc')