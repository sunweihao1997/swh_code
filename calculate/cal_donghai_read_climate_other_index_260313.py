import pandas as pd
import numpy as np
import sys

df = pd.read_csv(
    '/home/sun/data/download_data/climate_index_national_climate_center/M_other_index_260303.txt',
    sep=r'\s+',
    index_col=0
)

df.index.name = 'date'
df.columns.name = 'climate_index'

print(df.index.name)
print("=================================")
print(df.columns.name)

# 时间索引
df.index = pd.to_datetime(df.index.astype(str), format='%Y%m')

# 缺测值
df = df.replace(-999, np.nan)

print("================================= calculate the rolling mean by 3 months =================================")

# 严格3个月平均，但不要直接 dropna()
#print(f"shape of df is {df.shape}")
df_rolling = df.rolling(window=3, min_periods=3).mean()
#print(f"shape of df_rolling is {df_rolling.shape}")
#print(df_rolling.head(10))


# 只去掉前两行
df_rolling = df_rolling.iloc[2:]

# 或者改成：
# df_rolling = df_rolling.dropna(how='all')

month_letters = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']

def make_3mon_label(start_month):
    return ''.join(month_letters[(start_month - 1 + i) % 12] for i in range(3))

end_time = df_rolling.index
start_time = end_time - pd.DateOffset(months=2)

meta = pd.DataFrame({
    'start_time': start_time,
    'end_time': end_time,
    'season': [make_3mon_label(m) for m in start_time.month]
}, index=df_rolling.index)


result = pd.concat([meta, df_rolling], axis=1)
#
#print(result.head(20))
result.to_csv(
    '/home/sun/wd_14/data/data/download_data/climate_index_national_climate_center/other_index_3mon.csv',
    encoding='utf-8-sig',
    index=True,
    index_label='date'
)