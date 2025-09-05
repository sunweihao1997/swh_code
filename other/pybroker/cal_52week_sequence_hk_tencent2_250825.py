'''
2025-8-25
This script is to calculate the 52-week sequence for a given stock symbol for HK stock.

Note: Running on Tencent2 server.
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from scipy import stats
import random
import time

end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=52*7)).strftime("%Y%m%d")

# =============== Screening Stocks ===============
# Following should be updated weekly to get the latest stock list
current_date = datetime.now().strftime("%Y%m%d") ; print(f"Current Date: {current_date}")

stock_hk_spot_em_df = ak.stock_hk_main_board_spot_em()

stock_hk_spot_em_df.to_excel(f"/home/ubuntu/stock_data/all_stock_realtime/HK_real_time_{current_date}.xlsx", index=False)

spot_df = pd.read_excel(f"/home/ubuntu/stock_data/all_stock_realtime/HK_real_time_{current_date}.xlsx", dtype={"代码": str})

# 按照“成交量”降序排序
sorted_df = spot_df.sort_values(by="成交量", ascending=False)

# 取前1000个
top1000_df = sorted_df.head(1000).copy()


percentile_list = []

for index, row in top1000_df.iterrows():
    code = row['代码']
    name = row['名称']

    try:
        df = ak.stock_hk_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")

        if len(df) < 180:
            print(f"Skipping {code} ({name}) due to insufficient data.")
            percentile_list.append(None)
            continue

        percentile = stats.percentileofscore(df['收盘'], df['收盘'].iloc[-1], kind='rank')
        print(f"{code} ({name}) - 52-week percentile: {percentile:.2f}%")
        percentile_list.append(percentile)

    except Exception as e:
        print(f"Error processing {code} ({name}): {e}")
        percentile_list.append(None)

    time.sleep(random.randint(1, 20))

print("ALL DONE!")

top1000_df['52周百分位'] = percentile_list
top1000_df = top1000_df.dropna(subset=['52周百分位'])

today_str = datetime.today().strftime('%Y%m%d')
file_path = f"/home/ubuntu/stock_data/stock_percentile/HK_52week_percentile_output_{current_date}.xlsx"
top1000_df.to_excel(file_path, index=False)