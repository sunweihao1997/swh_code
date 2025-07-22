'''
2025-7-15
This script is to screen out some stocks and calculate the indicators for them.
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import numpy as np

sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df

end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

df_a = cal_index_for_stock("002032", start_date, end_date)


# ------------ 1.  recent 10 days slope for CCI ------------------
#print(df_a['CCI'].tail(10))
df_a['CCI_slope'] = df_a['CCI'].rolling(window=10).apply(
    lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
    raw=True
)
#print(df_a['CCI_slope'].tail(10)) # Have checked and results are correct

# ------------ 2.  recent ratio of positive CCI among past 14 days -----------------
N_window = 10 ; threshold_cci = 0.6
cci_pos = (df_a['CCI'] > 0).astype(int) ; cci_neg = (df_a['CCI'] < 0).astype(int)
cci_pos_ratio = cci_pos.rolling(window=N_window).sum() / N_window

# ------------ 3.  Slope of OBV index -----------------------
df_a['OBV_slope'] = df_a['OBV'].rolling(window=7).apply(
    lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
    raw=True
)
# ------------ 4.  Ratio inwhich OBV larger than OBV30; using 10 days -----------------
df_a['OBV_M30']   = df_a['OBV'].rolling(window=30).mean()
obv_pos           = ((df_a['OBV'] - df_a['OBV_M30']) > 0).astype(int)
obv_pos_ratio = obv_pos.rolling(window=10).sum() / 10
#print(obv_pos_ratio)

# ------------ 5. Slope of DIFF_MACD ------------
df_a['macd_diff_slope'] = df_a['MACDh_12_26_9'].rolling(window=5).apply(
    lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
    raw=True
)

# ------------ 6. Slope of MA60 --------------
df_a['MA60'] = df['收盘'].rolling(window=55).mean()
df_a['MA60_slope'] = df_a['MA60'].rolling(window=5).apply(
    lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
    raw=True
)

# ------------ 6. Slope of trix-diff --------------
df_a['trix_diff_slope'] = df_a['trix_diff'].rolling(window=5).apply(
    lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
    raw=True
)