'''
2024-6-5
This script is to calculate the correlation between the LSTC index and onset-date
'''
import xarray as xr
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import linregress

def partial_corr(A, B, C):
    # 计算 A 对 C 的回归
    slope_A, intercept_A, _, _, _ = linregress(C, A)
    A_residuals = A - (slope_A * C + intercept_A)
    
    # 计算 B 对 C 的回归
    slope_B, intercept_B, _, _, _ = linregress(C, B)
    B_residuals = B - (slope_B * C + intercept_B)

    #print(B_residuals)
    
    # 计算 A_residuals 和 B_residuals 之间的相关系数
#    partial_corr_coefficient = np.corrcoef(A, B_residuals)[0, 1]
    a, b = pearsonr(A, B_residuals)

    print(a)
    print(b)
    
    return a

f0 = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc")

f0_March = f0.sel(time=f0.time.dt.month.isin([3]))
f0_April = f0.sel(time=f0.time.dt.month.isin([4]))
#print(f0_March)

# data about onset early/late years
onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years.nc")
onset_day_file_42 = onset_day_file.sel(year=slice(1980, 2021)) #42 years

#print(onset_day_file_42)
correlation_coefficient, p_value = pearsonr(onset_day_file_42['onset_day'], f0_April['LSTC_sp_IOB'])

print(correlation_coefficient)

#partial_corr(onset_day_file_42['onset_day'], f0_April['OLR_mari_Afri'], f0_April['LSTC_psl_IOB'])
partial_corr(onset_day_file_42['onset_day'], f0_April['LSTC_sp_IOB'], f0_April['OLR_mari_Afri'], )
#print(a)