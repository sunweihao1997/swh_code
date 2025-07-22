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

sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df

# Get current status
#spot_df = ak.stock_zh_a_spot_em()
#spot_df.to_excel("/home/sun/data/other/real_time_250710.xlsx", index=False)
spot_df = pd.read_excel("/home/sun/data/other/real_time_250710.xlsx", dtype={"代码": str})

columns = spot_df.columns.tolist()

# 自动识别市盈率列名
pe_candidates = ["市盈率-动态", "市盈率(动)", "市盈率", "市盈率_TTM"]
pe_col = next((col for col in pe_candidates if col in columns), None)

# 把市盈率数据拉出来然后清除小于0的
spot_df = spot_df[["代码", "名称", pe_col]].dropna()
spot_df = spot_df[(spot_df[pe_col] > 0) & (spot_df[pe_col] < 50)]
    
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

    df_b = standardize_and_normalize(df_a)
    df_c = map_df(df_b[0])

    df_c.to_csv(output_file, index=False)
    print(f"Successfully processed {code} - {name}")

