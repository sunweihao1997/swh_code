'''
20240616
This script is to calculate the deviation of the index LSTC and OLR
'''
import xarray as xr
import numpy as np

# Original data
f0 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc")

#print(f0)
# Here only two index is select
lstc = f0['LSTC_psl_IOB'].data
olr  = f0['OLR_mari_Afri'].data

print(olr.shape)