'''
2023-9-24
This script calculate composite of the TRMM precipitation in early/late and climate onset years
'''
import xarray as xr
import numpy as np
import os

path0 = '/home/sun/mydown/trmm_precipitation/'
file_list = os.listdir(path0) ; file_list.sort()

# ========= Here I need to read onset date ===================
date_file = xr.open_dataset('/home/sun/data/onset_day_data/onsetdate.nc').sel(year=slice(1998, 2019))
date      = date_file['bob_onset_date'].data

# ========= Here I use whole date array to distinguish early and late years =========
date_file0 = xr.open_dataset('/home/sun/data/onset_day_data/onsetdate.nc')
date0      = date_file['bob_onset_date'].data
# Distinguish
early_year = np.array([])
late_year  = np.array([])
#print(date_file0)
for yyyy in range(2019-1998+1):
    if date[yyyy] > np.average(date) + np.std(date):
        late_year = np.append(late_year, yyyy+1998)
    elif date[yyyy] < np.average(date) - np.std(date):
        early_year = np.append(early_year, yyyy+1998)
    else:
        continue
#print(early_year)
# ========= Now calculate composite in different situation ===========================
# -30 0 +9 Total 40 days
ref_file = xr.open_dataset(path0 + file_list[5])
#print(ref_file) #Shape(1440, 400) lon, lat
composite_climate = np.zeros((40, 400, 1440))
composite_early   = np.zeros((40, 400, 1440))
composite_late    = np.zeros((40, 400, 1440))

# === Year loop for calculation ===
for yyyy in range(1998, 2020):
    # === Get subset of this year ===
    file_subset = []
    for ffff in file_list:
        if '.' + str(yyyy) in ffff:
            file_subset.append(ffff)

    #print(file_subset)
    # Get the onset day
    onset_day = date[yyyy - 1998]
    print('Now it is dealing with year {} to calculate climate average, the onset day is {}'.format(yyyy, onset_day))
    
    for dddd in range(-30, 9):
        # === First calculate climate average ===
        day_file = onset_day + dddd - 1 # position as the file location
        file0 = xr.open_dataset(path0 + file_subset[day_file])

        # exchange dimension among latitude and longitude
        trmm_data = np.transpose(file0['precipitation'].data, (1, 0))

        composite_climate[dddd + 30] += trmm_data / len(date)
        # Detect whether is early year
        if yyyy in early_year:
            print('Detect early year which is {}, Now calculate this year variable'.format(yyyy))
            composite_early[dddd + 30] += trmm_data / len(early_year)
        elif yyyy in late_year:
            print('Detect late year which is {}, Now calculate this year variable'.format(yyyy))
            composite_late[dddd + 30] += trmm_data / len(late_year)

    # Now finish calculation

# ======== Write to the nc file ============
ncfile  =  xr.Dataset(
{
    "onset_day": (["year"], date),
    "onset_day_early": (["year_early"], date[(early_year-1998).astype(int)]),
    "onset_day_late": (["year_late"],   date[(late_year-1998).astype(int)]),
    "trmm_climate":(["composite_day", "lat", "lon"], composite_climate),
    "trmm_early":  (["composite_day", "lat", "lon"], composite_early),
    "trmm_late":   (["composite_day", "lat", "lon"], composite_late),
},
coords={
    "year": (["year"], date_file['year'].data),
    "year_early": (["year_early"], early_year.astype(int)),
    "year_late":  (["year_late"],  late_year.astype(int)),
    "composite_day":(["composite_day"], np.linspace(-30, 9, 40)),
    "lat":(["lat"], ref_file['lat'].data),
    "lon":(["lon"], ref_file['lon'].data)
},
)
ncfile.attrs['description']  =  'This file save the composite TRMM precipitation during the monsoon onset in climate year, early year and late years. The unit is mm. Creation date is 20230925.'
ncfile.to_netcdf("/home/sun/data/composite/trmm_composite_1998_2019_onset_climate_early_late_year.nc")

    


    #print(composite_climate[30, :, 30])

