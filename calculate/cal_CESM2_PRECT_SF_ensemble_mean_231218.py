'''
2023-12-18
This script is to calculate the member-average among the CESM2 SF experiment, variable is PRECT
'''
import xarray as xr
import numpy as np
import os

# ======================== File information =================================

path_src = '/home/sun/CMIP6/LE_SF_CESM2/CESM2_LE_SF_PRECT/Aggregated/'

list_src = os.listdir(path_src)
#print(len(list_src))

ref_file = xr.open_dataset(path_src + list_src[0])
time     = ref_file.time.data
lat      = ref_file.lat.data
lon      = ref_file.lon.data

#print(ref_file.time.data)
shape    = ref_file['PRECT'].data.shape # 1980, 192, 288, time: 165 years from 1850 to 2014-12

# ===========================================================================

# ======================= Calculation =======================================

mean_prect = np.zeros(shape, dtype=np.float32)

member_num = len(list_src)

for i in range(member_num):
    f_member = xr.open_dataset(path_src + list_src[i])

    mean_prect += ( f_member['PRECT'].data / member_num )

# ===========================================================================

# ======================= Write to ncfile ===================================

ncfile  =  xr.Dataset(
{
    "PRECT": (["time", "lat", "lon"], mean_prect),
},
coords={
    "time": (["time"], time),
    "lat":  (["lat"],  lat),
    "lon":  (["lon"],  lon),
},
)

ncfile['PRECT'].attrs = ref_file['PRECT'].attrs

#
out_path = '/home/sun/CMIP6/LE_SF_CESM2/CESM2_LE_SF_PRECT/ensemble_mean/'
ncfile.to_netcdf(out_path + 'CESM2_LE_SF_PRECT_20_member_average.nc')
