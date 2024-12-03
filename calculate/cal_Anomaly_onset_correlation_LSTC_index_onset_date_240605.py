'''
2024-6-5
This script is to calculate the correlation between the LSTC index and onset-date
'''
import xarray as xr
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


f0 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc")

f0_Feb   = f0.sel(time=f0.time.dt.month.isin([2]))
f0_March = f0.sel(time=f0.time.dt.month.isin([3]))
f0_April = f0.sel(time=f0.time.dt.month.isin([4]))

f0_April['LSTC_psl_IOB'].data  = (f0_March['LSTC_psl_IOB'].data  + f0_April['LSTC_psl_IOB'].data)/2
f0_April['OLR_mari_Afri'].data = (f0_March['OLR_mari_Afri'].data + f0_April['OLR_mari_Afri'].data)/2
#print(f0_March)

# data about onset early/late years
onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years.nc")
onset_day_file_42 = onset_day_file.sel(year=slice(1980, 2021)) #42 years

#print(onset_day_file_42)
correlation_coefficient, p_value = pearsonr(onset_day_file_42['onset_day'], f0_April['OLR_mari_Afri'])

#print(correlation_coefficient)
#print(p_value)


# =================== try to regression ======================
#X_train, X_test, y_train, y_test = train_test_split(np.transpose(np.array([(f0_March['OLR_mari_Afri'].data - np.average(f0_March['OLR_mari_Afri'].data))/np.std(f0_March['OLR_mari_Afri'].data), (f0_March['LSTC_psl_IOB'].data - np.average(f0_March['LSTC_psl_IOB'].data))/np.std(f0_March['LSTC_psl_IOB'].data)])), onset_day_file_42['onset_day'].data, test_size=0.2, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(np.transpose(np.array([(f0_April['OLR_mari_Afri'].data - np.average(f0_April['OLR_mari_Afri'].data))/np.std(f0_April['OLR_mari_Afri'].data), (f0_April['LSTC_psl_IOB'].data - np.average(f0_April['LSTC_psl_IOB'].data))/np.std(f0_April['LSTC_psl_IOB'].data)])), (onset_day_file_42['onset_day'].data-np.average(onset_day_file_42['onset_day'].data))/np.std(onset_day_file_42['onset_day'].data), test_size=0.05, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(np.transpose(np.array([(f0_April['OLR_mari_Afri'].data - np.average(f0_April['OLR_mari_Afri'].data))/np.std(f0_April['OLR_mari_Afri'].data)])), (onset_day_file_42['onset_day'].data-np.average(onset_day_file_42['onset_day'].data))/np.std(onset_day_file_42['onset_day'].data), test_size=0.05, random_state=42)

a = (f0_April['OLR_mari_Afri'].data - np.average(f0_April['OLR_mari_Afri'].data))/np.std(f0_April['OLR_mari_Afri'].data)
b = (f0_April['LSTC_psl_IOB'].data - np.average(f0_April['LSTC_psl_IOB'].data))/np.std(f0_April['LSTC_psl_IOB'].data)
#a = (f0_March['OLR_mari_Afri'].data - np.average(f0_March['OLR_mari_Afri'].data))/np.std(f0_March['OLR_mari_Afri'].data)
#b = (f0_March['LSTC_psl_IOB'].data  - np.average(f0_March['LSTC_psl_IOB'].data)) /np.std(f0_March['LSTC_psl_IOB'].data)

#X_train, X_test, y_train, y_test = train_test_split(a[:, np.newaxis], onset_day_file_42['onset_day'].data, test_size=0.1, random_state=42)
# 创建线性回归模型并训练
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 输出回归系数和截距
print("回归系数:", model.coef_)
print("截距:", model.intercept_)

# 计算并输出模型性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("均方误差 (MSE):", mse)
print("R^2 值:", r2)

new_array = -0.66551099*a + 0.30323297*b + -0.033739785897145956
correlation_coefficient, p_value = pearsonr(onset_day_file_42['onset_day'], new_array)

#print(correlation_coefficient)
#print(p_value)
#
#print(onset_day_file_42['onset_day'])
#print(new_array)
#
#new_array1 = new_array.copy()
#new_array[(new_array1 - np.average(new_array1)) > 0] = 1
#
#new_array[(new_array1 - np.average(new_array1)) < 0] = -1
#
#dates = onset_day_file_42['onset_day'].data.copy()
#dates[(onset_day_file_42['onset_day'].data - np.average(onset_day_file_42['onset_day'].data)) > 0] = 1
#dates[(onset_day_file_42['onset_day'].data - np.average(onset_day_file_42['onset_day'].data)) < 0] = -1

std_dates = (onset_day_file_42['onset_day'] - np.average(onset_day_file_42['onset_day']))/np.std(onset_day_file_42['onset_day'])
std_index = new_array

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.scatter(std_index, std_dates, color='blue', alpha=0.5, label='Data points')
plt.plot([-2, 2], [0, 0], 'grey', linestyle='--', lw=2)
plt.plot([0,  0], [-2, 2], 'grey', linestyle='--', lw=2)
plt.title(' ')
plt.xlabel('index (March)')
plt.ylabel('Onset dates')
#plt.legend()
plt.xlim((-2, 2))
plt.ylim((-2, 2))

#plt.grid(True)
plt.savefig('/home/sun/paint/phd/phd_c5_fig11_scatter.pdf')