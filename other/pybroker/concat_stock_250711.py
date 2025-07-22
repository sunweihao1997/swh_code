import pandas as pd
import glob
import numpy as np

# 指定文件路径，例如合并同一文件夹下所有csv
file_paths = glob.glob('/home/sun/wd_14/data/data/other/stock_clean_train_v12/*.csv')

# 读取所有csv文件为DataFrame列表
#for file in file_paths:
#    a = pd.read_csv(file)
#    b = np.max(a['future_5_15_vs_past_3_return'])
#    if abs(b) < 1:
#        continue
#    else:
#        print(f"Max value in {file}: {b}")

df_list = [pd.read_csv(file) for file in file_paths]

# 使用concat纵向合并
combined_df = pd.concat(df_list, ignore_index=True)

# 保存到新csv文件
combined_df = combined_df[combined_df['future_12'].abs() < 0.8]
combined_df = combined_df[combined_df['close'] > 0.0]
combined_df.to_csv('/home/sun/wd_14/data/data/other/for_train_datasets_v12_250721.csv', index=False)
print(np.max(combined_df['future_12']))
print(len(combined_df))