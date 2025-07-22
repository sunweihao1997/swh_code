'''
2025-7-1
This script is to calculate the Mei-yu dates corresponding to the BOBSM onset
'''
import pandas as pd
import numpy as np
import xarray as xr
from scipy import stats
from scipy.stats import pearsonr
from scipy import signal

# Try to read txt file
df_changjiang = pd.read_csv("/home/sun/wd_14/data_beijing/meiyu/changjiang.csv")
df_jiangnan   = pd.read_csv("/home/sun/wd_14/data_beijing/meiyu/jiangnan.csv")
df_jianghuai  = pd.read_csv("/home/sun/wd_14/data_beijing/meiyu/jianghuai.csv")

changjiang_rumei_a = df_changjiang['Diff2-4'] ; changjiang_chumei_a = df_changjiang['Diff3-5'] ; changjiang_period_a = df_changjiang['(2~3)-(4~5)']
jiangnan_rumei_a   = df_jiangnan['Diff2-4']   ; jiangnan_chumei_a = df_jiangnan['Diff3-5']     ; jiangnan_period_a = df_jiangnan['(2~3)-(4~5)']

jianghuai_rumei_a  = df_jianghuai['Diff2-4']  ; jianghuai_chumei_a = df_jianghuai['Diff3-5']   ; jianghuai_period_a = df_jianghuai['(2~3)-(4~5)']

all_variable = [signal.detrend(changjiang_rumei_a), signal.detrend(changjiang_chumei_a), signal.detrend(changjiang_period_a),
                signal.detrend(jiangnan_rumei_a),  signal.detrend(jiangnan_chumei_a),  signal.detrend(jiangnan_period_a),
                signal.detrend(jianghuai_rumei_a), signal.detrend(jianghuai_chumei_a), signal.detrend(jianghuai_period_a)]

onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")

# Calculate the BOBSM onset date anomaly
onset_day_anomaly = onset_day_file['onset_day'].data - np.average(onset_day_file['onset_day'].data)

# Plotting the correlation between BOBSM onset date and Mei-yu dates
import matplotlib.pyplot as plt

#plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
#plt.rcParams['axes.unicode_minus'] = False




fig, axes = plt.subplots(3, 3, figsize=(20, 20))

column_title = ['Ru-Mei', 'Chu-Mei', 'Mei-yu Length',]
row_title    = ['Changjiang', 'Jiangnan', 'JiangHuai']

num = 0
for j in range(3):
    for i in range(3):
        ax = axes[i, j]

        x = onset_day_anomaly ; y = all_variable[num]

        # 计算线性回归拟合
        coefficients = np.polyfit(x, y, 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        y_fit = slope * x + intercept

        # 计算皮尔逊相关系数
        r, p = pearsonr(x, y)

        # 绘制散点图和拟合线
        ax.scatter(x, y, alpha=1)
        ax.plot(x, y_fit, color='red', alpha=0.8)

        # 在右上角添加皮尔逊相关系数
        ax.text(0.95, 0.95, f'r = {r:.2f}',
                horizontalalignment='right',
                verticalalignment='top',
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
        
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)

        ax.set_title(row_title[i], fontsize=12, color='blue', loc='left')
        ax.set_title(column_title[j], fontsize=12, color='blue', loc='right')

        num += 1
        
plt.savefig("/home/sun/paint/CD/BOBSM_onset_date_vs_meiyu_anomaly_detrend.png", dpi=700, bbox_inches='tight')