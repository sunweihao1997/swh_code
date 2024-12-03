'''
2024-6-15
This script is to preprocess the data, which for the regression calculating by ncl
'''
import xarray as xr
import numpy as np
import os

# ============== File Information ==================

index_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc")

# other files
f1         = xr.open_dataset("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_psl.nc")
f2         = xr.open_dataset("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_u10.nc")
f3         = xr.open_dataset("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_v10.nc")

#print(f1.time)
# ==================================================

# ======== 1. Deal with index_file =========

Feb_index_file = index_file.sel(time=index_file.time.dt.month.isin([2]))
Mar_index_file = index_file.sel(time=index_file.time.dt.month.isin([3]))
Apr_index_file = index_file.sel(time=index_file.time.dt.month.isin([4]))

#print(Apr_index_file.time)

# ======== 2. Deal with psl file =========

Feb_psl_file = f1.sel(time=f1.time.dt.month.isin([2]))
Mar_psl_file = f1.sel(time=f1.time.dt.month.isin([3]))
Apr_psl_file = f1.sel(time=f1.time.dt.month.isin([4]))

# ======== 3. Deal with u10 file =========

Feb_u10_file = f2.sel(time=f2.time.dt.month.isin([2]))
Mar_u10_file = f2.sel(time=f2.time.dt.month.isin([3]))
Apr_u10_file = f2.sel(time=f2.time.dt.month.isin([4]))

# ======== 4. Deal with v10 file =========

Feb_v10_file = f3.sel(time=f3.time.dt.month.isin([2]))
Mar_v10_file = f3.sel(time=f3.time.dt.month.isin([3]))
Apr_v10_file = f3.sel(time=f3.time.dt.month.isin([4]))

Feb_index_file.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_feb.nc")
Mar_index_file.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_mar.nc")
Apr_index_file.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_apr.nc")

#Feb_psl_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_psl_feb.nc")
#Mar_psl_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_psl_mar.nc")
#Apr_psl_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_psl_apr.nc")
#
#Feb_u10_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_u10_feb.nc")
#Mar_u10_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_u10_mar.nc")
#Apr_u10_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_u10_apr.nc")
#
#Feb_v10_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_v10_feb.nc")
#Mar_v10_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_v10_mar.nc")
#Apr_v10_file.to_netcdf("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_v10_apr.nc")

print("Done")