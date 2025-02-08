'''
2024-12-3
This script is to plot the time series of msl/u10/v10 of the composite average in early and late onset years

Only for the March-May period

v2 241205: add February
'''
import numpy as np
import xarray as xr


# Onset date file
date_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")
#print(date_file)

# Mask file
mask_file = xr.open_dataset("/home/sun/data/mask/ERA5_land_sea_mask.nc")

early_years = date_file.year_early.data
late_years  = date_file.year_late.data
#print(late_years)

file_path   = "/home/sun/mydown/ERA5/era5_psl_u10_v10_daily/"

# =========================== First Part. Deal with sea level pressure ========================
def land_sea_msl_contrast(f0, mask, land_range=[0, 20, 70, 90], ocean_range=[-20, 20, 40, 100]):
    '''
        This function calculate the land sea contrast using msl
    '''
    # 1. mask the value
    f0_land  = f0.sel(longitude=slice(land_range[2], land_range[3]), latitude=slice(land_range[1], land_range[0]))     ; mask_land  = mask.sel(longitude=slice(land_range[2], land_range[3]), latitude=slice(land_range[1], land_range[0]))
    f0_ocean = f0.sel(longitude=slice(ocean_range[2], ocean_range[3]), latitude=slice(ocean_range[1], ocean_range[0])) ; mask_ocean = mask.sel(longitude=slice(ocean_range[2], ocean_range[3]), latitude=slice(ocean_range[1], ocean_range[0]))

    time = f0.valid_time.data ; result_series = np.zeros((time.shape[0]))
    for tt in range(time.shape[0]):
        f0_land['msl'].data[tt][mask_land.lsm.data[0]<0.8]   = np.nan
        f0_ocean['msl'].data[tt][mask_ocean.lsm.data[0]>0.2] = np.nan

        result_series[tt] = np.nanmean(f0_ocean['msl'].data[tt]) - np.nanmean(f0_land['msl'].data[tt])

    return result_series

climate_msl_fmam = np.zeros((92 + 28)) ; early_msl_fmam = climate_msl_fmam.copy() ; late_msl_fmam = climate_msl_fmam.copy() # Land-Sea Contrast
climate_msl_all_fmam = np.zeros((42, 92 + 28))
early_msl_all_fmam = np.zeros((6, 92 + 28))  ;  late_msl_all_fmam = np.zeros((9, 92 + 28))  ;  count_early = 0  ;  count_late = 0

# March-May
# 1.1 Claim the array
climate_msl = np.zeros((92)) ; early_msl = climate_msl.copy() ; late_msl = climate_msl.copy() # Land-Sea Contrast
climate_msl_all = np.zeros((42, 92))
early_msl_all = np.zeros((6, 92))  ;  late_msl_all = np.zeros((9, 92))  ;  count_early = 0  ;  count_late = 0

for yyyy in range(1980, 2022): # 42 years
    f_msl   = xr.open_dataset(file_path + str(int(yyyy)) + ".nc")
    series0 = land_sea_msl_contrast(f_msl, mask_file)
    print(f'Now it is {yyyy}')

    if yyyy in late_years:

        late_msl += series0/9
        late_msl_all[count_late] = series0 ; count_late += 1

    elif yyyy in early_years:

        early_msl += series0/6
        early_msl_all[count_early] = series0 ; count_early += 1

    climate_msl += series0/42
    climate_msl_all[yyyy - 1980] = series0

climate_msl_fmam[28:]      = climate_msl
early_msl_fmam[28:]        = early_msl
late_msl_fmam[28:]         = late_msl
climate_msl_all_fmam[:, 28:] = climate_msl_all
early_msl_all_fmam[:, 28:] = early_msl_all
late_msl_all_fmam[:, 28:]  = late_msl_all

#sys.exit()

# Feb
# 1.1 Claim the array
climate_msl = np.zeros((28)) ; early_msl = climate_msl.copy() ; late_msl = climate_msl.copy() # Land-Sea Contrast
climate_msl_all = np.zeros((42, 28))
early_msl_all = np.zeros((6, 28))  ;  late_msl_all = np.zeros((9, 28))  ;  count_early = 0  ;  count_late = 0

for yyyy in range(1980, 2022): # 42 years
    f_msl   = xr.open_dataset(file_path + str(int(yyyy)) + "_Feb.nc")
    series0 = land_sea_msl_contrast(f_msl, mask_file)
    print(f'Now it is {yyyy}')
    if yyyy in late_years:

        late_msl += series0[:28]/9
        late_msl_all[count_late] = series0[:28] ; count_late += 1

    elif yyyy in early_years:

        early_msl += series0[:28]/6
        early_msl_all[count_early] = series0[:28] ; count_early += 1

    climate_msl_all[yyyy - 1980] = series0[:28]
    climate_msl += series0[:28]/42

climate_msl_fmam[:28]      = climate_msl
climate_msl_all_fmam[:, :28]  = climate_msl_all
early_msl_fmam[:28]        = early_msl
late_msl_fmam[:28]         = late_msl
early_msl_all_fmam[:, :28] = early_msl_all
late_msl_all_fmam[:,  :28]  = late_msl_all

# 1.2 Save to the file
ncfile  =  xr.Dataset(
{
    "climate_msl": (["time"], climate_msl_fmam),
    "early_msl":   (["time"], early_msl_fmam),
    "late_msl":    (["time"], late_msl_fmam),
    "climate_msl_all": (["year", "time"], climate_msl_all_fmam),
    "early_msl_all":   (["year_early", "time"], early_msl_all_fmam),
    "late_msl_all":    (["year_late",  "time"], late_msl_all_fmam),
},
coords={
    "time": (["time"], np.linspace(1, 28 + 92, 28 + 92)),
    "year_early": (["year_early"], early_years),
    "year_late":  (["year_late"],  late_years),
    "year":  (["year"],  np.linspace(1980, 2021, 42)),
},
)
ncfile.attrs['description']  =  'Land-Sea Thermal Contrast using msl. Calculation script is /home/sun/swh_code/calculate/cal_Anomaly_onset_time_series_msl_u10v10_v3_240124.py. 10 degree means ocean range is -10 to 10'
ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/ERA5_msl_land_sea_contrast_feb_may_daily_10degree.nc")