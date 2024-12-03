'''
2024-4-4
This script is to compare the MErra2 and ERA5 BOBSM onset date
'''
import xarray as xr
import numpy as np

fera = xr.open_dataset('/home/sun/data/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc').sel(year=slice(1980, 2019))
fmer = xr.open_dataset('/home/sun/data/onset_day_data/onsetdate.nc')

date_era = fera['onset_day'].data
date_mer = fmer['bob_onset_date'].data

import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
fig, ax = plt.subplots()
ax.plot(date_era, color='red')
ax.plot(date_mer, color='black')

ax.grid()

fig.savefig("test.png")

'''
Summary:
For the 1980-2019, although with difference in some years, it shows very similar characteristics between MERRA2 and ERA5
'''