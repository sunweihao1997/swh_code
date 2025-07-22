'''
2025-6-26
This script is to calculate the correlationship between BOBSM onset date and monthly precipitation
'''
import xarray as xr
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr


# Path for the precipitation data
precp_data = xr.open_dataset("/home/sun/data/download/ERA5_Precipitation_monthly_Apr_Sep/ERA5_precipitation_monthly_Apr_Sep_1980_2025.nc")

# Path to the BOBSM onset date
onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")

new_lat = np.arange(90, -90.1, -1.0) ; print(new_lat.shape)
new_lon = np.arange(0, 360, 1.0) ; print(new_lon.shape)

precp_data_low = precp_data.interp(latitude=new_lat, longitude=new_lon)
#print(precp_data)
# Investigate the time-axis of precip data
# print(precp_data.valid_time.data) # 4-9 6 months per year

# ---- Calculate the correlation between BOBSM and 5-9 monthly precipitation ----
# slice data
period_data = precp_data_low.sel(valid_time=slice("1980-05-01", "2021-10-31"))
may_data    = period_data.sel(valid_time=period_data.valid_time.dt.month == 5)
jun_data    = period_data.sel(valid_time=period_data.valid_time.dt.month == 6)
jul_data    = period_data.sel(valid_time=period_data.valid_time.dt.month == 7)
aug_data    = period_data.sel(valid_time=period_data.valid_time.dt.month == 8)
sep_data    = period_data.sel(valid_time=period_data.valid_time.dt.month == 9)

# calculating the correlation
correlation_may = np.zeros((181, 360)) ; p_may = np.zeros((181, 360))
correlation_jun = np.zeros((181, 360)) ; p_jun = np.zeros((181, 360))
correlation_jul = np.zeros((181, 360)) ; p_jul = np.zeros((181, 360))
correlation_aug = np.zeros((181, 360)) ; p_aug = np.zeros((181, 360))
correlation_sep = np.zeros((181, 360)) ; p_sep = np.zeros((181, 360))

scorrelation_may = np.zeros((181, 360)) ; sp_may = np.zeros((181, 360))
scorrelation_jun = np.zeros((181, 360)) ; sp_jun = np.zeros((181, 360))
scorrelation_jul = np.zeros((181, 360)) ; sp_jul = np.zeros((181, 360))
scorrelation_aug = np.zeros((181, 360)) ; sp_aug = np.zeros((181, 360))
scorrelation_sep = np.zeros((181, 360)) ; sp_sep = np.zeros((181, 360))

#print(jul_data)

for i in range(181):
    for j in range(360):
        r, p_value = pearsonr(onset_day_file['onset_day'].data, may_data['tp'].data[:, i, j])
        correlation_may[i, j] = r ; p_may[i, j] = p_value
        rho, p_value = spearmanr(onset_day_file['onset_day'].data, may_data['tp'].data[:, i, j])
        scorrelation_may[i, j] = rho ; sp_may[i, j] = p_value

        r, p_value = pearsonr(onset_day_file['onset_day'].data, jun_data['tp'].data[:, i, j])
        correlation_jun[i, j] = r ; p_jun[i, j] = p_value
        rho, p_value = spearmanr(onset_day_file['onset_day'].data, jun_data['tp'].data[:, i, j])
        scorrelation_jun[i, j] = rho ; sp_jun[i, j] = p_value

        r, p_value = pearsonr(onset_day_file['onset_day'].data, jul_data['tp'].data[:, i, j])
        correlation_jul[i, j] = r ; p_jul[i, j] = p_value
        rho, p_value = spearmanr(onset_day_file['onset_day'].data, jul_data['tp'].data[:, i, j])
        scorrelation_jul[i, j] = rho ; sp_jul[i, j] = p_value

        r, p_value = pearsonr(onset_day_file['onset_day'].data, aug_data['tp'].data[:, i, j])
        correlation_aug[i, j] = r ; p_aug[i, j] = p_value
        rho, p_value = spearmanr(onset_day_file['onset_day'].data, aug_data['tp'].data[:, i, j])
        scorrelation_aug[i, j] = rho ; sp_aug[i, j] = p_value

        r, p_value = pearsonr(onset_day_file['onset_day'].data, sep_data['tp'].data[:, i, j])
        correlation_sep[i, j] = r ; p_sep[i, j] = p_value
        rho, p_value = spearmanr(onset_day_file['onset_day'].data, sep_data['tp'].data[:, i, j])
        scorrelation_sep[i, j] = rho ; sp_sep[i, j] = p_value

ncfile  =  xr.Dataset(
            {
                "correlation_jun": (["latitude", "longitude"], correlation_jun),  
                "correlation_jul": (["latitude", "longitude"], correlation_jul),
                "correlation_aug": (["latitude", "longitude"], correlation_aug),
                "correlation_sep": (["latitude", "longitude"], correlation_sep),
                "correlation_may": (["latitude", "longitude"], correlation_may),
                "scorrelation_jun": (["latitude", "longitude"], scorrelation_jun),
                "scorrelation_jul": (["latitude", "longitude"], scorrelation_jul),
                "scorrelation_aug": (["latitude", "longitude"], scorrelation_aug),
                "scorrelation_sep": (["latitude", "longitude"], scorrelation_sep),
                "scorrelation_may": (["latitude", "longitude"], scorrelation_may),

                "p_jun": (["latitude", "longitude"], p_jun),  
                "p_jul": (["latitude", "longitude"], p_jul),
                "p_aug": (["latitude", "longitude"], p_aug),
                "p_sep": (["latitude", "longitude"], p_sep),
                "p_may": (["latitude", "longitude"], p_may),
                "sp_jun": (["latitude", "longitude"], sp_jun),
                "sp_jul": (["latitude", "longitude"], sp_jul),
                "sp_aug": (["latitude", "longitude"], sp_aug),
                "sp_sep": (["latitude", "longitude"], sp_sep),
                "sp_may": (["latitude", "longitude"], sp_may),
            },
            coords={
                "latitude":  (["latitude"],    precp_data_low.latitude.data),
                "longitude":  (["longitude"],  precp_data_low.longitude.data),
            },
        )

ncfile.attrs['description'] = 'Created on 2025-6-27 by /home/sun/swh_code/calculate/cal_Anomaly_onset_correlation_between_BOBSM_single_month_precipitation_250626.py on Huaibei N100'
ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_correlation_BOBSMonset_monthly_precipitation_lowresolution.nc")