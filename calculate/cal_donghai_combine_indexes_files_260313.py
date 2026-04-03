import pandas as pd

# 读取
df_atm = pd.read_csv("/home/sun/wd_14/data/data/download_data/climate_index_national_climate_center/atmosphere_index_3mon.csv", index_col=0)
df_ocn = pd.read_csv("/home/sun/wd_14/data/data/download_data/climate_index_national_climate_center/ocean_index_3mon.csv", index_col=0)
df_oth = pd.read_csv("/home/sun/wd_14/data/data/download_data/climate_index_national_climate_center/other_index_3mon.csv", index_col=0)

# 如果索引是日期，可转成 datetime
df_atm.index = pd.to_datetime(df_atm.index)
df_ocn.index = pd.to_datetime(df_ocn.index)
df_oth.index = pd.to_datetime(df_oth.index)

# 横向合并（按索引对齐）
df_all = pd.concat([df_atm, df_ocn, df_oth], axis=1)

# 保存
df_all.to_csv("/home/sun/wd_14/data/data/download_data/climate_index_national_climate_center/all_index_3mon.csv")

print(df_all.head())
print(df_all.shape)