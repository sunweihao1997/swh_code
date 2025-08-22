'''
2025-8-3
This script is to calculate the 52-week sequence for a given stock symbol.
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from scipy import stats

end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=52*7)).strftime("%Y%m%d")

# =============== Screening Stocks ===============

spot_df = pd.read_excel("/home/sun/data/other/real_time_250710.xlsx", dtype={"代码": str})

columns = spot_df.columns.tolist()

# 自动识别市盈率列名
pe_candidates = ["市盈率-动态", "市盈率(动)", "市盈率", "市盈率_TTM", "总市值"]
pe_col = next((col for col in pe_candidates if col in columns), None)

# 把市盈率数据拉出来然后清除小于0的
spot_df = spot_df[["代码", "名称", "总市值", pe_col]].dropna()
spot_df = spot_df[(spot_df[pe_col] > 0) & (spot_df[pe_col] < 100)]

# 筛选出来100e市值以上的
spot_df = spot_df[spot_df['总市值'] > 5e10]

percentile_list = []

for index, row in spot_df.iterrows():
    code = row['代码']
    name = row['名称']

    try:
        df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")

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

spot_df['52周百分位'] = percentile_list
spot_df = spot_df.dropna(subset=['52周百分位'])

today_str = datetime.today().strftime('%Y%m%d')
file_path = f"/home/sun/data/other/stock_percentile/52week_percentile_output_{today_str}.xlsx"
spot_df.to_excel(file_path, index=False)