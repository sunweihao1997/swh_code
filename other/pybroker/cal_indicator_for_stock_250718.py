'''
2025-7-10
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

# Get current status
#spot_df = ak.stock_zh_a_spot_em()
#spot_df.to_excel("/home/sun/data/other/real_time_250710.xlsx", index=False)
spot_df = pd.read_excel("/home/sun/data/other/real_time_250710.xlsx", dtype={"代码": str})

columns = spot_df.columns.tolist()

# 自动识别市盈率列名
pe_candidates = ["市盈率-动态", "市盈率(动)", "市盈率", "市盈率_TTM",]
pe_col = next((col for col in pe_candidates if col in columns), None)

# 把市盈率数据拉出来然后清除小于0的
spot_df = spot_df[["代码", "名称",  "总市值", pe_col]].dropna()
spot_df = spot_df[(spot_df[pe_col] > 0) & (spot_df[pe_col] < 50)]

# 筛选出来50e市值以上的
spot_df = spot_df[spot_df['总市值'] > 5e9]

    
# 日期范围 recent 5 years
end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

# =========================== Start Calculation ============================
#print(spot_df.iloc[5]['代码'])
#print(start_date)
#test_a = cal_index_for_stock(spot_df.iloc[5]['代码'], start_date, end_date)
#
#test_b = standardize_and_normalize(test_a)
#test_c = map_df(test_b[0])
#test_c.to_csv("/home/sun/data/other/test.csv", index=False)

out_path = "/home/sun/wd_14/data/data/other/stock_indicator/"
for index, row in spot_df.iterrows():
    code = row['代码']
    name = row['名称']
    output_file = os.path.join(out_path, f"{code}.csv")
    
    if os.path.exists(output_file):   
        print(f"File {output_file} already exists, skipping...")
        continue

    print(f"Processing {code} - {name}...")
    
    df_a = cal_index_for_stock(code, start_date, end_date)
    
    if df_a is None:
        print(f"Insufficient data for {code}, skipping...")
        continue

    # ------------ 1.  recent 10 days slope for CCI ------------------
    #print(df_a['CCI'].tail(10))
    df_a['CCI_slope'] = df_a['CCI'].rolling(window=10).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )
    df_a['CCI_slope20'] = df_a['CCI'].rolling(window=20).apply(
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
    df_a['OBV_slope20'] = df_a['OBV'].rolling(window=20).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )
    # ------------ 4.  Ratio inwhich OBV larger than OBV30; using 10 days -----------------
    df_a['OBV_M30']   = df_a['OBV'].rolling(window=30).mean()
    obv_pos           = ((df_a['OBV'] - df_a['OBV_M30']) > 0).astype(int)
    obv_pos_ratio = obv_pos.rolling(window=10).sum() / 10
    df_a['OBV_ratio'] = obv_pos_ratio
    #print(obv_pos_ratio)

    # ------------ 5. Slope of DIFF_MACD ------------
    df_a['macd_diff_slope'] = df_a['MACDh_12_26_9'].rolling(window=5).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )
    df_a['macd_diff_slope20'] = df_a['MACDh_12_26_9'].rolling(window=20).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )
    # ------------ 6. Slope of MA60 --------------
    df_a['MA60'] = df_a['收盘'].rolling(window=55).mean()
    df_a['MA60_slope'] = df_a['MA60'].rolling(window=3).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )

    # ------------ 6. Slope of trix-diff --------------
    df_a['trix_diff_slope'] = df_a['trix_diff'].rolling(window=5).apply(
        lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
        raw=True
    )

    df_b = standardize_and_normalize(df_a)
    df_c = map_df(df_b[0])

    df_c.to_csv(output_file, index=False)
    print(f"Successfully processed {code} - {name}")

