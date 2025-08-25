'''
2025-7-22
This script is to calculate the index for stock monitoring
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import numpy as np

sys.path.append("/home/ubuntu/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df


def cal_base_index(code, start_date, end_date):
    # 1. Get the fundamental data
    df_a = cal_index_for_stock(code, start_date, end_date)

    if df_a is None:
        return None
    elif len(df_a) < 365:
        return None

    # 2. Slope of the 10-day CCI
    df_a['CCI_slope10'] = df_a['CCI'].rolling(window=10).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )

    df_a['CCI_slope5'] = df_a['CCI'].rolling(window=5).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )

    # 3. Ratio of the positive CCI in recent 7 days
    N_window = 7
    cci_pos = (df_a['CCI'] > 0).astype(int) ; cci_neg = (df_a['CCI'] < 0).astype(int)
    cci_pos_ratio10 = cci_pos.rolling(window=N_window).sum() / N_window
    df_a['CCI_pos_ratio10'] = cci_pos_ratio10

    # 4. Ratio of the positive CCI in recent 21 days
    N_window = 21
    cci_pos_ratio21 = cci_pos.rolling(window=N_window).sum() / N_window
    df_a['CCI_pos_ratio21'] = cci_pos_ratio21

    # 4. Ratio of the positive CCI in recent 30 days
    N_window = 30
    cci_pos_ratio30 = cci_pos.rolling(window=N_window).sum() / N_window
    df_a['CCI_pos_ratio30'] = cci_pos_ratio30

    # 5. Whether the ratio of positive CCI is increasing
    df_a['CCI_pos_ratio7_gt_21_30'] = (
        (df_a['CCI_pos_ratio10'] > df_a['CCI_pos_ratio21'])
    ).astype(int)

    # 6. Slope of the OBV index
    df_a['OBV_slope'] = df_a['OBV'].rolling(window=7).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )
    df_a['OBV_slope20'] = df_a['OBV'].rolling(window=20).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )
    df_a['OBV_M30']   = df_a['OBV'].rolling(window=30).mean()
    obv_pos           = ((df_a['OBV'] - df_a['OBV_M30']) > 0).astype(int)
    obv_pos_ratio = obv_pos.rolling(window=10).sum() / 10
    df_a['OBV_ratio'] = obv_pos_ratio

    # 7. Slow OBV slope is positive
    df_a['OBV_slope20_pos'] = (df_a['OBV_slope20'] > 0).astype(int)

    # 8. Slow MACD slope is positive
    df_a['macd_diff_slope'] = df_a['MACDh_12_26_9'].rolling(window=3).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )

    # ------------ 9. Slope of MA60 --------------
    df_a['MA60'] = df_a['收盘'].rolling(window=55).mean()
    df_a['MA60_slope'] = df_a['MA60'].rolling(window=3).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )
    df_a['MA60_slope_pos'] = (
        (df_a['MA60_slope'] > 0)
    ).astype(int)

    # 10. Whether MA5 larger than MA60
    df_a['MA5'] = df_a['收盘'].rolling(window=5).mean()
    df_a['MA5_MA60'] = (df_a['MA5'] > df_a['MA60']).astype(int)

    # 11. Slope of trix-diff
    df_a['trix_diff_slope'] = df_a['trix_diff'].rolling(window=3).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )

    # 12. Slope of rsi
    df_a['rsi_slope'] = df_a['rsi'].rolling(window=7).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )

    # 13. Ratio of rsi larger than 50 in recent 14 days
    rsi_pos = (df_a['rsi'] > 50).astype(int)
    rsi_pos_ratio = rsi_pos.rolling(window=14).sum() / 14
    df_a['rsi_pos_ratio'] = rsi_pos_ratio

    # 14. Whether upward moving in close regarding to the MA60
    df_a["cross_up"] = (
        (df_a["收盘"] > df_a["MA60"]) &
        (df_a["收盘"].shift(1) <= df_a["MA60"].shift(1))
    )

    df_a["cross_last7"] = (
        df_a["cross_up"]
        .rolling(window=7, min_periods=1)
        .max()
        .astype(bool)
    )

    return df_a