'''
2023-2-10
This script calculate the abnormal years (early or late) onset day using ERA5 data
'''
import xarray as xr
import numpy as np

f0  =  xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc').sel(year=slice(1980, 2021))

days  =  f0['onset_day'].data
years =  f0['year'].data

print(np.average(days)) # 124.7
print(np.std(days)) # 12
#print(years)

early_year  =  np.array([], dtype=int)
late_year   =  np.array([], dtype=int)

early_day   =  np.array([], dtype=int)
late_day    =  np.array([], dtype=int)

for i in range(years.shape[0]):
    if days[i] < (np.average(days) - np.std(days)):
        early_year = np.append(early_year, years[i])
        early_day  = np.append(early_day, days[i])
    elif days[i] > (np.average(days) + np.std(days)):
        late_year  = np.append(late_year, years[i])
        late_day   = np.append(late_day, days[i])
    else:
        continue

#print(early_year) ; print(early_day)
#print(late_year)  ; print(late_day)

# ----------- save to the ncfile ------------------
ncfile  =  xr.Dataset(
{
    "onset_day": (["year"], days),
    "onset_day_early": (["year_early"], early_day),
    "onset_day_late": (["year_late"], late_day),
},
coords={
    "year": (["year"], years),
    "year_early": (["year_early"], early_year),
    "year_late":  (["year_late"],  late_year),
},
)
ncfile.attrs['description']  =  'one standard deviation from the mean when choosing a standard. The orginal onset date is derived from /home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc. Relevant script is /home/sun/mycode_copy/calculate/cal_abnormal_onset_year_ERA5_230210.py'
ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")