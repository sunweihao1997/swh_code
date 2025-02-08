'''
2025-1-25
This script is to paint the scatter plot which show the relation between LSTC and onset date
'''
import xarray as xr
import numpy as np
from scipy.stats import pearsonr


# =============== Read the file =================
onset_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")
lstc_file  = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc")

lstc_file_april = lstc_file.sel(time=lstc_file.time.dt.month.isin([4]))


# =============== 2. Calculation ===============
# 2.1 calculate the monthly mean
lstc_a = (lstc_file_april["LSTC_psl_IOB"].data - np.average(lstc_file_april["LSTC_psl_IOB"].data)) - np.std(lstc_file_april["LSTC_psl_IOB"].data)


# 2.2.4 onset data standardization
onset_data = (onset_file["onset_day"].data - np.average(onset_file["onset_day"].data)) / np.std(onset_file["onset_day"].data)
#print(lstc_ma)


# ============== 3. Plotting ====================
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))  # 设置图表大小
scatter = plt.scatter(-1 * lstc_a, onset_data,)

plt.savefig("test.png")

# ============== 4. output correlation ============
corr_coefficient, p_value = pearsonr(lstc_a, onset_data)

print(f"Pearson 相关系数: {corr_coefficient:.2f}")
print(f"p 值: {p_value:.4f}")