'''
2025-7-21
This script serves as an example regarding monitoring the stock data preparation process.
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import xgboost as xgb

# 加载保存的模型
model = xgb.XGBClassifier()
model.load_model('/home/sun/xgboost_model_v2.json')
print("模型已加载！")


sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df

code = "001215"
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")
end_date = datetime.today().strftime("%Y%m%d")

# Example of calculating indicators for a specific stock

# Special 

df_a = cal_index_for_stock(code, start_date, end_date)
df_b = standardize_and_normalize(df_a)
df_c = map_df(df_b[0])

features = [
    'rsi', 'OBV_ratio', 'CCI_slope_z', 'OBV_slope_z', 'macd_diff_slope_z',
    'MA60_slope_z', 'trix_diff_slope_z', 'CCI', 'macd_diff_slope20_z', 'CCI_slope20_z', 'OBV_slope20_z'
]
df_x = df_c[features]

y_pred = model.predict(df_x)
df_x['prediction'] = y_pred
df_x['date'] = df_c['date']

df_x.to_csv(f"/home/sun/wd_14/data/data/other/prediction_{code}.csv", index=False)
