'''
2025-1-25
This script is to paint the scatter plot which show the relation between LSTC and onset date
'''
import xarray as xr
import numpy as np
from scipy.stats import pearsonr
import sys


# =============== Read the file =================
onset_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")
lstc_file  = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/ERA5_msl_land_sea_contrast_feb_may_daily_10degree.nc")

#print(lstc_file['year_early'].data)
#print(lstc_file["climate_msl_all"].data[4])
#print(lstc_file["early_msl_all"].data[0])
#sys.exit()



# =============== 2. Calculation ===============
# 2.1 calculate the monthly mean
# 2.2.1 Feb-April
lstc_fma = (np.average(lstc_file["climate_msl_all"].data[:, :(28 + 31 + 30)], axis=1) - np.average(np.average(lstc_file["climate_msl_all"].data[:, :(28 + 31 + 30)], axis=1))) / np.std(np.average(lstc_file["climate_msl_all"].data[:, :(28 + 31 + 30)], axis=1))
# 2.2.2 March-May
lstc_mam = (np.average(lstc_file["climate_msl_all"].data[:, 28 : (28 + 31 + 30 + 31)], axis=1) - np.average(np.average(lstc_file["climate_msl_all"].data[:, 28 : (28 + 31 + 30 + 31)], axis=1))) / np.std(np.average(lstc_file["climate_msl_all"].data[:, 28 : (28 + 31 + 30 + 31)], axis=1))
# 2.2.3 March-April
lstc_ma  = (np.average(lstc_file["climate_msl_all"].data[:, 28 : (28 + 31 + 30)], axis=1) - np.average(np.average(lstc_file["climate_msl_all"].data[:, 28 : (28 + 31 + 30)], axis=1))) / np.std(np.average(lstc_file["climate_msl_all"].data[:, 28 : (28 + 31 + 30)], axis=1)) # March-April
# 2.2.4 April
lstc_a   = (np.average(lstc_file["climate_msl_all"].data[:, 28 + 31 : (28 + 31 + 30)], axis=1) - np.average(np.average(lstc_file["climate_msl_all"].data[:, 28 + 31 : (28 + 31 + 30)], axis=1))) / np.std(np.average(lstc_file["climate_msl_all"].data[:, 28 + 31 : (28 + 31 + 30)], axis=1)) # March-April
# 2.2.5 April-March
lstc_am  = (np.average(lstc_file["climate_msl_all"].data[:, 28 + 31 : (28 + 31 + 30 + 31)], axis=1) - np.average(np.average(lstc_file["climate_msl_all"].data[:, 28 + 31 : (28 + 31 + 30 + 31)], axis=1))) / np.std(np.average(lstc_file["climate_msl_all"].data[:, 28 + 31 : (28 + 31 + 30 + 31)], axis=1)) # March-April


# 2.2.4 onset data standardization
onset_data = (onset_file["onset_day"].data - np.average(onset_file["onset_day"].data)) / np.std(onset_file["onset_day"].data)
#print(lstc_ma)


# ============== 3. Plotting ====================
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))  # 设置图表大小
scatter = plt.scatter(-1 * lstc_am, onset_data,)

plt.savefig("test.png")

# ============== 4. output correlation ============
corr_coefficient, p_value = pearsonr(lstc_a, onset_data)

print(f"Pearson 相关系数: {corr_coefficient:.2f}")
print(f"p 值: {p_value:.4f}")