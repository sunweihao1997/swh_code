import xarray as xr
import numpy as np
import pandas as pd

nc_path = "/home/sun/wd_14/data_beijing/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years.nc"  # 修改为你的文件路径

# 读取数据
ds = xr.open_dataset(nc_path)

# 1) 只保留 1980 年及以后
year_vals = ds["year"].values
mask_1980 = year_vals >= 1980
ds80 = ds.isel(year=mask_1980)

# 2) 取出 onset_day 与 year，并忽略 NaN
years = ds80["year"].values
onset = ds80["onset_day"].values.astype(float)

valid = np.isfinite(onset)
years_v = years[valid]
onset_v = onset[valid]

if onset_v.size == 0:
    raise ValueError("1980年及以后 onset_day 全是 NaN，无法计算均值/标准差。")

# 3) 计算均值与标准差（这里用总体标准差 ddof=0；如需样本标准差改为 ddof=1）
mu = onset_v.mean()
sigma = onset_v.std(ddof=0)

lower = mu - sigma
upper = mu + sigma

early_mask = onset_v < lower
late_mask  = onset_v > upper

early_years = years_v[early_mask]
early_onset = onset_v[early_mask]

late_years = years_v[late_mask]
late_onset = onset_v[late_mask]

# 4) 输出 early / late 年所包含的数据及对应年份
early_df = pd.DataFrame({"year": early_years.astype(int), "onset_day": early_onset}).sort_values("year")
late_df  = pd.DataFrame({"year": late_years.astype(int),  "onset_day": late_onset}).sort_values("year")

print("=== 1980年及以后统计 ===")
print(f"mean = {mu:.4f}, std = {sigma:.4f}")
print(f"early 阈值: onset_day < {lower:.4f}")
print(f"late  阈值: onset_day > {upper:.4f}\n")

print("=== Early years (onset_day < mean-std) ===")
print(early_df.to_string(index=False) if len(early_df) else "无")

print("\n=== Late years (onset_day > mean+std) ===")
print(late_df.to_string(index=False) if len(late_df) else "无")

# 可选：保存到 CSV
out_csv = "onset_day_outside_1std_after1980.csv"
pd.concat(
    [early_df.assign(category="early"), late_df.assign(category="late")],
    ignore_index=True
).to_csv(out_csv, index=False, encoding="utf-8-sig")
print(f"\n已写出: {out_csv}")

# （可选）如果你还想看原文件里“已经存好的”year_early/year_late，也可以过滤到1980年以后输出：
if "year_early" in ds and "onset_day_early" in ds:
    ex_early = pd.DataFrame({
        "year": ds["year_early"].values.astype(int),
        "onset_day_early": ds["onset_day_early"].values.astype(float)
    })
    ex_early = ex_early[ex_early["year"] >= 1980].sort_values("year")
    print("\n=== 文件内置的 early 列表（过滤到1980年后）===")
    print(ex_early.to_string(index=False) if len(ex_early) else "无")

if "year_late" in ds and "onset_day_late" in ds:
    ex_late = pd.DataFrame({
        "year": ds["year_late"].values.astype(int),
        "onset_day_late": ds["onset_day_late"].values.astype(float)
    })
    ex_late = ex_late[ex_late["year"] >= 1980].sort_values("year")
    print("\n=== 文件内置的 late 列表（过滤到1980年后）===")
    print(ex_late.to_string(index=False) if len(ex_late) else "无")
