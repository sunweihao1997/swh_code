'''
2025-7-11
This script is to map Chinese column names to English for stock data.
'''
import pandas as pd
import os

import sys
sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df

# Get file list
file_all = os.listdir("/home/sun/wd_14/data/data/other/stock_indicator/")
for i in file_all:
    if not i.endswith(".csv"):
        continue
#    print(i)
    df = pd.read_csv(f"/home/sun/wd_14/data/data/other/stock_indicator/{i}")

    
    # Map columns
    df_mapped = map_df(df)
#    print(df_mapped.columns)
    
    # Save to new file
    df_mapped.to_csv(f"/home/sun/wd_14/data/data/other/stock_indicator_mapped/{i}", index=False)

    print(f"Mapped {i} and saved to stock_indicator_mapped directory.")