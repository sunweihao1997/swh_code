'''
2025-7-22
Calculate correlation between Nino34 index and hutai index
'''
import pandas as pd
import numpy as np
import xarray as xr

# Read file
df_nino34 = xr.open_dataset('/home/sun/wd_14/data/data/download_data/nino34/nino34.long.anom.nc')
df_hutai  = pd.read_excel('/home/sun/wd_14/data/data/download_data/nino34/浒苔面积时间统计(更新至2024年).xlsx')
#print(df_hutai.columns)

ds_nino_0924 = df_nino34.sel(time=slice("2009-01-01", "2024-12-31"))

correlations_nc = np.array([])
correlations_nt = np.array([])

def smooth_convolve(data):
    kernel = np.array([1, 1, 1]) / 3
    return np.convolve(data, kernel, mode='same')  # 会自动处理边界

#ds_nino_0924['value'].data = smooth_convolve(ds_nino_0924['value'].data)

for i in range(1, 8):
    # Extract the Nino34 index for the month of i
    nino34_month = ds_nino_0924.sel(time=ds_nino_0924['time'].dt.month == i)
    
    nino_index   = nino34_month['value'].data

    cover_area        = df_hutai['最大覆盖面积 /km2'].values[1:-1]
    distribution_area = df_hutai['最大分布面积 /km2'].values[1:-1]

    #print(cover_area)

    # detrend the series
    nino_index_detrended = nino_index - np.polyval(np.polyfit(np.arange(len(nino_index)), nino_index, 1), np.arange(len(nino_index)))
    cover_area_detrended = cover_area - np.polyval(np.polyfit(np.arange(len(cover_area)), cover_area, 1), np.arange(len(cover_area)))
    distribution_area_detrended = distribution_area - np.polyval(np.polyfit(np.arange(len(distribution_area)), distribution_area, 1), np.arange(len(distribution_area)))
#    nino_index_detrended = nino_index
#    cover_area_detrended = cover_area
#    distribution_area_detrended = distribution_area


    # 计算相关系数矩阵
    corr_nc = np.corrcoef(nino_index_detrended, cover_area_detrended)
    corr_nt = np.corrcoef(nino_index_detrended, distribution_area_detrended)

    # 提取相关系数
    corr_nc0 = corr_nc[0, 1]
    corr_nt0 = corr_nt[0, 1]

    correlations_nc = np.append(correlations_nc, corr_nc0)
    correlations_nt = np.append(correlations_nt, corr_nt0)

print("Nino34 and hutai correlation coefficients for each month:")
print("Nino34 and hutai cover area correlation:", correlations_nc)
print("Nino34 and hutai distribution area correlation:", correlations_nt)

    