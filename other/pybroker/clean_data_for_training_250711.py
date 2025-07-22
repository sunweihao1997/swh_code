'''
This script is used to clean and prepare stock data for training a model.
'''
import pandas as pd
import os

path1 = "/home/sun/wd_14/data/data/other/stock_12_return/"
path2 = "/home/sun/wd_14/data/data/other/stock_indicator/"

print('yes')

file1_list = os.listdir(path1)
file2_list = os.listdir(path2)

for file1 in file1_list:
    if not file1.endswith(".xlsx"):
        continue
    
    else:
        print(f"Processing {file1}...")
        df1 = pd.read_excel(os.path.join(path1, file1))

        if os.path.exists(os.path.join(path2, file1.replace("xlsx", "csv"))):
            df2 = pd.read_csv(os.path.join(path2, file1.replace("xlsx", "csv")))
        else:
            print(f"File {file1.replace('xlsx', 'csv')} not exists in {path2}, skipping...")
            continue

        df2['future_12'] = df1['return_12']
        print('yes')

        df2 = df2.dropna()
        df2 = df2[['code', 'date', 'close', 'CCI', 'rsi', 'CCI_slope_z', 'OBV_slope_z', 'OBV_ratio', 'macd_diff_slope_z', 'MA60_slope_z', 'trix_diff_slope_z', 'macd_diff_slope20_z', 'CCI_slope20_z', 'OBV_slope20_z', 'future_12']]
        df2.to_csv("/home/sun/wd_14/data/data/other/stock_clean_train_v12/" + file1.replace("xlsx", "csv"), index=False)

