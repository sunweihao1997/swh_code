'''
2025-6-28
Calculate the composite anomaly of precipitation and 850hPa wind for the onset day of BOBSM
'''
import xarray as xr
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy import stats

# Path for the precipitation data
data_path = "/home/sun/wd_14/download_reanalysis/ERA5/monthly_pressure_0.5_0.5/"

onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")

precp_data = xr.open_dataset("/home/sun/data/download/ERA5_Precipitation_monthly_Apr_Sep/ERA5_precipitation_monthly_Apr_Sep_1980_2025.nc")

# Read date data
onset_day = onset_day_file['onset_day'].data ;  onset_day_early = onset_day_file['onset_day_early'].data ; onset_day_late = onset_day_file['onset_day_late'].data

# Read precipitation data for information
ref_file = xr.open_dataset(data_path + "ERA5_monnthly_pressure.0.5x0.5.1997.nc")

# Interpolate the precipitation data to the same grid as the ERA5 pressure data
precp_data_low = precp_data.interp(latitude=ref_file.latitude, longitude=ref_file.longitude)
precp_data_low = precp_data_low.sel(valid_time=slice("1980-05-01", "2021-05-31"))
precp_data_low = precp_data_low.sel(valid_time=precp_data_low.valid_time.dt.month.isin([5,]))

# Claim the array
u850_all = np.zeros((len(onset_day), 361, 720)) ; u850_early = np.zeros((len(onset_day_early), 361, 720)) ; u850_late = np.zeros((len(onset_day_late), 361, 720))
v850_all = np.zeros((len(onset_day), 361, 720)) ; v850_early = np.zeros((len(onset_day_early), 361, 720)) ; v850_late = np.zeros((len(onset_day_late), 361, 720))
pr_all   = np.zeros((len(onset_day), 361, 720)) ; pr_early = np.zeros((len(onset_day_early), 361, 720))   ; pr_late = np.zeros((len(onset_day_late), 361, 720))


year0 = 1980 ; num_early = 0 ; num_late = 0
for i in range(len(onset_day)):
    # Read the monthly data
    f_single = xr.open_dataset(data_path + f"ERA5_monnthly_pressure.0.5x0.5.{year0 + i}.nc").sel(pressure_level=850)

    # Save data into the all array
    u850_all[i, :, :] = f_single['u'].data[4] ; v850_all[i, :, :] = f_single['v'].data[4] ; pr_all[i, :, :] = precp_data_low['tp'].data[i, :, :]

    if onset_day[i] in onset_day_early:
        u850_early[num_early, :, :] = f_single['u'].data[4] ; v850_early[num_early, :, :] = f_single['v'].data[4] ; pr_early[num_early, :, :] = precp_data_low['tp'].data[i, :, :]
        num_early += 1
    elif onset_day[i] in onset_day_late:
        u850_late[num_late, :, :] = f_single['u'].data[4] ; v850_late[num_late, :, :] = f_single['v'].data[4] ; pr_late[num_late, :, :] = precp_data_low['tp'].data[i, :, :]
        num_late += 1

# Calculate the students t-test for the early and late onset
p_early = np.zeros((361, 720)) ; p_late = np.zeros((361, 720))
for i in range(361):
    for j in range(720):
        t_stat, p_value = stats.ttest_ind(pr_all[:, i, j], pr_early[:, i, j], equal_var=False)
        p_early[i, j] = p_value

        t_stat, p_value = stats.ttest_ind(pr_all[:, i, j], pr_late[:, i, j], equal_var=False)
        p_late[i, j] = p_value

# Save into the array
ncfile  =  xr.Dataset(
            {
                "u850_climate": (["latitude", "longitude"], np.average(u850_all, axis=0)), 
                "u850_early":   (["latitude", "longitude"], np.average(u850_early, axis=0)), 
                "u850_late":    (["latitude", "longitude"], np.average(u850_late, axis=0)),  
                "v850_climate": (["latitude", "longitude"], np.average(v850_all, axis=0)),
                "v850_early":   (["latitude", "longitude"], np.average(v850_early, axis=0)),
                "v850_late":    (["latitude", "longitude"], np.average(v850_late, axis=0)),
                "pr_climate": (["latitude", "longitude"], np.average(pr_all, axis=0)),
                "pr_early":   (["latitude", "longitude"], np.average(pr_early, axis=0)),
                "pr_late":    (["latitude", "longitude"], np.average(pr_late, axis=0)),
                "p_early": (["latitude", "longitude"], p_early),
                "p_late": (["latitude", "longitude"], p_late),
            },
            coords={
                "latitude":  (["latitude"],    precp_data_low.latitude.data),
                "longitude":  (["longitude"],  precp_data_low.longitude.data),
            },
        )

ncfile.attrs['description'] = 'Created on 2025-6-29 by /home/sun/swh_code/calculate/cal_Anomaly_Onset_composite_anomaly_precip_850wind_250628.py on Huaibei N100'
ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_composite_anomaly_precip_850wind_onset_day_early_late.nc", mode='w', format='NETCDF4')
