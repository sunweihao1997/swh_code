# -*- coding: utf-8 -*-
"""
时滞相关分析：佘山 vs 崇明（盐度）
- 读取：sheshan_new.xlsx；wuhaogou_chongming.xlsx(表：05454崇明)
- 对齐时间、去缺测
- 计算时滞相关（崇明滞后 & 佘山滞后）：12~144 小时
- 输出：CSV + 可选绘图
"""

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# 0) 参数
# -----------------------------
FILE_SHESHAN = "/mnt/f/数据_东海局/salinity/sheshan_new.xlsx"
FILE_CHONGMING = "/mnt/f/数据_东海局/salinity/wuhaogou_chongming.xlsx"
SHEET_CHONGMING = "05454崇明"
LAGS = [12, 24, 48, 72, 96, 120, 144]   # 单位：小时
OUTPUT_DIR = Path("./")

# -----------------------------
# 1) 读取数据
# -----------------------------
df_sheshan = pd.read_excel(FILE_SHESHAN)               # 列：时间、盐度（以及站点信息）
df_chongming = pd.read_excel(FILE_CHONGMING, sheet_name=SHEET_CHONGMING)  # 列：日期、盐度

# -----------------------------
# 2) 清洗 & 对齐
# -----------------------------
# 仅保留时间与盐度列，并转换时间为 datetime，设为索引
ss = (df_sheshan[["时间", "盐度"]]
      .rename(columns={"时间": "time", "盐度": "盐度_佘山"})
     )
ss["time"] = pd.to_datetime(ss["time"])
ss = ss.set_index("time").sort_index()

cm = (df_chongming[["日期", "盐度"]]
      .rename(columns={"日期": "time", "盐度": "盐度_崇明"})
     )
cm["time"] = pd.to_datetime(cm["time"])
cm = cm.set_index("time").sort_index()

# 如果存在重复时间戳，可以按小时聚合平均（可按需启用）
# ss = ss.groupby(pd.Grouper(freq="H")).mean()
# cm = cm.groupby(pd.Grouper(freq="H")).mean()

# 按时间交集对齐（只保留双方都有的数据点）
df = ss.join(cm, how="inner")

# 剔除缺测
df = df.dropna()

# 断言一下是按小时的数据（若不是，可自行 resample 到小时级）
# print(df.index.inferred_freq)

# 取出两列序列
s_ss = df["盐度_佘山"]
s_cm = df["盐度_崇明"]

# -----------------------------
# 3) 定义：某方向滞后的相关与显著性
# -----------------------------
def lag_corr_and_pval(base_series: pd.Series, lagged_series: pd.Series, lag_hours: int):
    """
    计算：lagged_series 相对 base_series 滞后 lag_hours 小时时的相关系数与 p 值。
    正的 lag_hours 表示 lagged_series 向后 shift(lag_hours)。
    返回：r, p, n(有效样本数)
    """
    shifted = lagged_series.shift(lag_hours)
    valid = pd.concat([base_series, shifted], axis=1).dropna()
    if len(valid) < 3:   # pearson 至少需要 3 个点
        return float("nan"), float("nan"), len(valid)
    r, p = stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
    return r, p, len(valid)

# -----------------------------
# 4) 计算两种方向的时滞相关
# -----------------------------
rows_cm_lag = []   # 崇明滞后（s_cm 相对 s_ss 滞后）
for h in LAGS:
    r, p, n = lag_corr_and_pval(s_ss, s_cm, h)  # base=佘山；滞后=崇明
    rows_cm_lag.append({"滞后小时(崇明滞后)": h, "r": r, "p_value": p, "有效样本数": n})

rows_ss_lag = []   # 佘山滞后（s_ss 相对 s_cm 滞后）
for h in LAGS:
    r, p, n = lag_corr_and_pval(s_cm, s_ss, h)  # base=崇明；滞后=佘山
    rows_ss_lag.append({"滞后小时(佘山滞后)": h, "r": r, "p_value": p, "有效样本数": n})

df_cm_lag = pd.DataFrame(rows_cm_lag)
df_ss_lag = pd.DataFrame(rows_ss_lag)

# 保存结果
out1 = OUTPUT_DIR / "lag_corr_chongming_lag.csv"
out2 = OUTPUT_DIR / "lag_corr_sheshan_lag.csv"
df_cm_lag.to_csv(out1, index=False, encoding="utf-8-sig")
df_ss_lag.to_csv(out2, index=False, encoding="utf-8-sig")

print("=== 崇明滞后（相对佘山） ===")
print(df_cm_lag.round(6))
print("\n=== 佘山滞后（相对崇明） ===")
print(df_ss_lag.round(6))
print(f"\n结果已保存：\n- {out1}\n- {out2}")

# -----------------------------
# 5) 可选：绘制滞后相关曲线（两方向同图）
# -----------------------------
try:
    plt.figure(figsize=(7,5))
    plt.plot(df_cm_lag["滞后小时(崇明滞后)"], df_cm_lag["r"], marker="o", label="崇明滞后")
    plt.plot(df_ss_lag["滞后小时(佘山滞后)"], df_ss_lag["r"], marker="s", label="佘山滞后")
    plt.xlabel("time lag (hours)")
    plt.ylabel("correlation r")
    #plt.title("佘山 vs 崇明 盐度：时滞相关曲线")
    plt.grid(True, linestyle="--", alpha=0.5)
    #plt.legend()
    plt.tight_layout()
    plt.savefig("/mnt/f/wsl_plot/donghai/salinity/lag_corr_sheshan_chongming.png", dpi=300)
except Exception as e:
    print("绘图失败（可忽略，仅与本地 matplotlib 字体/环境相关）：", e)
