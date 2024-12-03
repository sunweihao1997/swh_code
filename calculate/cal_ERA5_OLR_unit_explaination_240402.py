'''
2024-4-2
This script is to see the unit of the ERA5 OLR variable. Because its unit is J m**-2, it is important to see how to contert it into W m**-2
'''
import xarray as xr
import numpy as np

ref_file = xr.open_dataset('/home/sun/data/other_data/down_ERA5_hourly_OLR_convert_float/1963_hourly_OLR.nc')

data0    = ref_file['ttr'].data[0:48, 20, 20]

#print(np.sum(data0[:24]/(24*3600)))
#print(np.average(data0[:24])/(3600))

# Short summary:
# Its unit is J m**-2 for one hour, which could be converted into W m**-2 by dividing 3600s (1hour)