'''
2024-4-2
This script calculate the correlation between onset date and the OLR/LSTC index for the 1979-2021

This is to see whether the period selection can affect the correlationship
'''
import xarray as xr
import numpy as np
from scipy import stats
from scipy import signal

# ========== Read the correlated files =======================
lstc = xr.open_dataset('/home/sun/data/long_time_series_after_process/ERA5/ERA5_pentad_LSTC_1940_2022.nc').sel(year=slice(1959, 1980))
olr  = xr.open_dataset('/home/sun/data/long_time_series_after_process/ERA5/ERA5_pentad_OLR_maritime_continent_1940_2022.nc').sel(year=slice(1959, 1980))
nino = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_SST_Nino34_pentad_month_1959_2021.nc').sel(year=slice(1959, 1980))

onset_date = xr.open_dataset('/home/sun/data/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc').sel(year=slice(1959, 1980))
#print(np.average(onset_date['onset_day'].data))
# =============================================================

# ========== Calculate the correlation ========================
coor_lstc = np.zeros(73)
coor_olr  = np.zeros(73)
coor_nino = np.zeros(73)
corre_lstc_olr = np.zeros(73)
for i in range(73):
    #corre_index_lstc   = stats.pearsonr(onset_date['onset_day'].data - np.average(onset_date['onset_day'].data), lstc['LSTC'].data[:, i] - np.average(lstc['LSTC'].data[:, i]))
    #corre_index_olr    = stats.pearsonr(onset_date['onset_day'].data - np.average(onset_date['onset_day'].data), olr['olr'].data[:, i]   - np.average(olr['olr'].data[:, i]))
    corre_index_lstc   = stats.pearsonr(signal.detrend(onset_date['onset_day'].data) - np.average(signal.detrend(onset_date['onset_day'].data)), signal.detrend(lstc['LSTC'].data[:, i]) - np.average(signal.detrend(lstc['LSTC'].data[:, i])))
    corre_index_olr    = stats.pearsonr(signal.detrend(onset_date['onset_day'].data) - np.average(signal.detrend(onset_date['onset_day'].data)), signal.detrend(olr['olr'].data[:, i])   - np.average(signal.detrend(olr['olr'].data[:, i])))
    corre_index_nino   = stats.pearsonr(signal.detrend(onset_date['onset_day'].data) - np.average(signal.detrend(onset_date['onset_day'].data)), signal.detrend(nino['pentad_nino34'].data[:, i])   - np.average(signal.detrend(nino['pentad_nino34'].data[:, i])))
    corre_index_olr_lstc  = stats.pearsonr(signal.detrend(lstc['LSTC'].data[:, i]) - np.average(signal.detrend(lstc['LSTC'].data[:, i])), signal.detrend(olr['olr'].data[:, i])   - np.average(signal.detrend(olr['olr'].data[:, i])))

    coor_lstc[i] = corre_index_lstc[0]
    coor_olr[i]  = corre_index_olr[0]
    coor_nino[i] = corre_index_nino[0]
    corre_lstc_olr[i] = corre_index_olr_lstc[0]

ncfile  =  xr.Dataset(
                    {
                        'onset_with_lstc': (["time",], coor_lstc),
                        'onset_with_olr': (["time",], coor_olr),
                        'onset_with_nino':(["time",], coor_nino),
                        'lstc_with_olr': (['time',], corre_lstc_olr),
                    },
                    coords={
                        "time": (["time"], np.linspace(1,73,73)),
                    },
                    )
ncfile.attrs['description'] = 'created on 2024-4-2. This file includes the correlation between onset date and OLR/LSTC index. The script is cal_correlation_onsetdate_pentad_LSTC_OLR_period1979-2021_240402.py on ubuntu(beijing). The period is 1979-2021'

ncfile.to_netcdf('/home/sun/data/ERA5_data_monsoon_onset/index/correlation/onset_dates_with_pentad_OLR_LSTC_detrend_new.nc')