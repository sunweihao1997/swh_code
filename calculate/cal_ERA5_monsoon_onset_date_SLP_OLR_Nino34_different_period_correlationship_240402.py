'''
2024-4-2
This script is to see the correlationship between monsoon onset date and Nino34 index
'''
import xarray as xr
import numpy as np
from scipy import stats
from scipy import signal

period1 = slice(1940, 1980)
period2 = slice(1980, 2010)

onset_date = xr.open_dataset('/home/sun/data/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc')
olr        = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_OLR_Feb-Mar-Apr_month_maritime_continent_1940-2020.nc')
slp        = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_SLP_Feb-Mar-Apr_land-sea-diff_70-90_1940-2020.nc')
olr2       = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_OLR_Mar-Apr_month_maritime_continent_1940-2020.nc')
slp2        = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_SLP_Mar_land-sea-diff_70-90_1940-2020.nc')

def cal_period_correlationship(period, onset_f, nino34_f, var):
    ''' month corresponds to the Nino34, month is the real month '''
    onset_date0 = onset_f.sel(year=period)
    nino340     = nino34_f.sel(year=period,)

    corre0      = stats.pearsonr(signal.detrend(onset_date0['onset_day'].data) - np.average(signal.detrend(onset_date0['onset_day'].data)), signal.detrend(nino340[var].data) - np.average(signal.detrend(nino340[var].data)))

    print(corre0)

cal_period_correlationship(period1, onset_date, slp, 'FMA_SLP')
#cal_period_correlationship(period2, onset_date, olr, 'FMA_OLR')

# Student t-test to see whether the onset-date is different under two period
#def cal_period_ttest(onset_f, perioda, periodb):
#    onset_datea = onset_f.sel(year=perioda)
#    onset_dateb = onset_f.sel(year=periodb)
#
#    t_stat, p_value = stats.ttest_ind(onset_datea['onset_day'].data, onset_dateb['onset_day'].data)
#
#    print(f'T统计量: {t_stat}, P值: {p_value}')
#
#cal_period_ttest(onset_date, period1, period2)