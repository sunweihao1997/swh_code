'''
This is to see the dates in onset early and late years
'''
import xarray as xr
import numpy as np

date_file = xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/onsetdate.nc')

early = []
late = []
year_early = []
year_late  = []

dates = date_file.bob_onset_date.data
for i in range(40):
    if dates[i] < np.average(dates) - np.std(dates):
        year_early.append(1980+i)
        early.append(dates[i])
    elif dates[i] > np.average(dates) + np.std(dates):
        year_late.append(1980+i)
        late.append(dates[i])

print(year_late)
print(np.average(early))
