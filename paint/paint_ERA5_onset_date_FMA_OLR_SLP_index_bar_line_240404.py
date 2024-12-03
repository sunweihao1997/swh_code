'''
2024-4-4
This script is overlap the monsoon onset date and SLP/OLR index averaged among Feb-Apr to display their correlationship
'''
import xarray as xr
import numpy as np
from scipy import stats
from scipy import signal

onset_date_file = xr.open_dataset('/home/sun/data/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc').sel(year=slice(1980, 2020))
SLP_file        = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_SLP_month_land_slp_70-90_1940-2020.nc').sel(year=slice(1980, 2020))
OLR_file        = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_OLR_month_maritime_continent_1940-2020.nc').sel(year=slice(1980, 2020))

# ==== Filter out onset early/normal/late years ====
onset_date      = onset_date_file['onset_day'].data
#print(np.average(onset_date))
onset_early     = np.array([])
onset_late      = np.array([])

onset_std       = np.std(onset_date)
#print(onset_std)
for yyyy in range(len(onset_date_file.year.data)):
    if abs(onset_date[yyyy] - np.average(onset_date)) > onset_std:
        if (onset_date[yyyy] - np.average(onset_date)) > 0:
            onset_late = np.append(onset_late, onset_date_file.year.data[yyyy])
        else:
            onset_early = np.append(onset_early, onset_date_file.year.data[yyyy])

    else:
        continue

'''
late: [1983. 1987. 1993. 1997. 2010. 2016. 2018. 2020.]
early:[1984. 1985. 1999. 2000. 2009. 2017.]
'''

# ========== Calculate the FMA average ================
FMA_SLP = np.average(SLP_file['FMA_SLP'].data[:, 2:5], axis=1)
FMA_OLR = np.average(OLR_file['FMA_OLR'].data[:, 1:5], axis=1)

fma_std = (FMA_SLP - np.average(FMA_SLP))/np.std(FMA_SLP)
olr_std = -1 * (FMA_OLR - np.average(FMA_OLR))/np.std(FMA_OLR)

#print(olr_std)

# Start painting
import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()

x        = np.linspace(1980, 2020, 41)

# PLot1
#ax1.bar(x, (onset_date_file['onset_day'] - np.average(onset_date_file['onset_day'])) / np.std(onset_date_file['onset_day']),)
ax1.bar(x, (onset_date_file['onset_day'] - np.average(onset_date_file['onset_day'])))

ax2 = ax1.twinx() 

ax2.plot(x, olr_std, color='blue')
ax2.plot(x, fma_std, color='red')

#plt.savefig('test.png')

# ============================================================================================================


#============================================= Multiple regression =========================================
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

X = sm.add_constant(np.column_stack((FMA_SLP - np.average(FMA_SLP), FMA_OLR - np.average(FMA_OLR))))

model = sm.OLS(onset_date_file['onset_day'].data - np.average(onset_date_file['onset_day']), X).fit()


#print(model.summary())
x3 = 0.0658 * (FMA_SLP) - 0.498 * (FMA_OLR)

ax2.plot(x, (x3 - np.average(x3))/np.std(x3), color='green')

plt.savefig('test.png')

# Scatter plot

fig, ax1 = plt.subplots()

ax1.scatter((x3 - np.average(x3))/np.std(x3), onset_date_file['onset_day'].data - np.average(onset_date_file['onset_day']))

ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-25, 25)

ax1.plot([-2.5, 2.5], [0, 0], 'k--', alpha=0.2)
ax1.plot([0, 0], [-25, 25], 'k--', alpha=0.2)

slope, intercept = np.polyfit((x3 - np.average(x3))/np.std(x3), onset_date_file['onset_day'].data - np.average(onset_date_file['onset_day']), 1)
y_reg            = slope * (x3 - np.average(x3))/np.std(x3) + intercept

# 绘制线性回归线
plt.plot((x3 - np.average(x3))/np.std(x3), y_reg, label='线性回归线', color='blue')

plt.savefig('test2.png')

# ========================================================================================================================

# Calculate partial correlation
# 首先，我们使用c来拟合b，并获取残差。这些残差代表了在控制c的情况下b的变化。
b_c_resid = sm.OLS(FMA_SLP, sm.add_constant(FMA_OLR)).fit().resid

# 然后，我们也使用c来拟合a，并获取残差。这些残差代表了在控制c的情况下a的变化。
a_c_resid = sm.OLS(onset_date_file['onset_day'].data, sm.add_constant(FMA_OLR)).fit().resid

# 最后，我们计算在控制了c之后，a和b残差之间的相关性，这就是偏相关系数。
partial_corr = np.corrcoef(a_c_resid, b_c_resid)[0, 1]

print(f'偏相关系数为: {partial_corr}')

b_c_resid = sm.OLS(FMA_OLR, sm.add_constant(FMA_SLP)).fit().resid

# 然后，我们也使用c来拟合a，并获取残差。这些残差代表了在控制c的情况下a的变化。
a_c_resid = sm.OLS(onset_date_file['onset_day'].data, sm.add_constant(FMA_SLP)).fit().resid

# 最后，我们计算在控制了c之后，a和b残差之间的相关性，这就是偏相关系数。
partial_corr = np.corrcoef(a_c_resid, b_c_resid)[0, 1]

print(f'偏相关系数为: {partial_corr}')