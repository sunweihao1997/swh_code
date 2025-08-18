'''
2025-8-6
This script is to calculate the HK stock indicators for a given stock symbol.
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
from scipy import stats

sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock_hk, standardize_and_normalize, map_df

end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=5*365)).strftime("%Y%m%d")

stock_hk_hist_df = ak.stock_hk_hist(symbol="00001", period="daily", start_date=start_date, end_date=end_date, adjust="qfq")

df_a = cal_index_for_stock_hk("00001", start_date, end_date)

print(df_a)
