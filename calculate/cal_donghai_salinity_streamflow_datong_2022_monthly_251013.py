'''
2025-10-13
This script calculates the monthly average streamflow for Donghai in 2022.
'''
import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# ========================= File Location ========================
INPUT_FILE = '/mnt/f/数据_东海局/salinity/大通径流量.xlsx'

f0         = pd.read_excel(INPUT_FILE)
f0['时间'] = pd.to_datetime(f0['时间'], errors='coerce')  # 兼容带/不带毫秒的字符串

jan, feb, mar, apr_, may, jun, jul, aug, sep, oct_, nov, dec = ([] for _ in range(12))

for idx, row in f0.iterrows():
    t  = row['时间']
    v  = row['流量']

    if pd.isna(t):
        continue  # 跳过无效时间
    if pd.isna(v):
        continue  # 跳过无效流量
    if v<100:
        continue  # 跳过异常时间

    if t.year == 2022:
        m = t.month
        if   m == 1:  jan.append(v)
        elif m == 2:  feb.append(v)
        elif m == 3:  mar.append(v)
        elif m == 4:  apr_.append(v)
        elif m == 5:  may.append(v)
        elif m == 6:  jun.append(v)
        elif m == 7:  jul.append(v)
        elif m == 8:  aug.append(v)
        elif m == 9:  sep.append(v)
        elif m == 10: oct_.append(v)
        elif m == 11: nov.append(v)
        elif m == 12: dec.append(v)

jan_avg = np.array(jan); feb_avg = np.array(feb); mar_avg = np.array(mar); apr_avg = np.array(apr_)
may_avg = np.array(may); jun_avg = np.array(jun); jul_avg = np.array(jul); aug_avg = np.array(aug)
sep_avg = np.array(sep); oct_avg = np.array(oct_); nov_avg = np.array(nov); dec_avg = np.array(dec)

# calculate monthly averages
def arr_mean(a): return float(np.mean(a)) if a.size else np.nan
monthly_mean_2022 = {
    1: arr_mean(jan_avg),  2: arr_mean(feb_avg),  3: arr_mean(mar_avg),  4: arr_mean(apr_avg),
    5: arr_mean(may_avg),  6: arr_mean(jun_avg),  7: arr_mean(jul_avg),  8: arr_mean(aug_avg),
    9: arr_mean(sep_avg), 10: arr_mean(oct_avg), 11: arr_mean(nov_avg), 12: arr_mean(dec_avg),
}
print(monthly_mean_2022)



