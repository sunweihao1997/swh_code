# -*- coding: utf-8 -*-
"""
在同一张图上叠加两组逐月径流量：
1) 第一组：柱状图
2) 第二组：折线图
3) 在柱状图上方叠加二者差值（柱状-折线）
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ---------- 数据（按你给的） ----------
group1 = [  # 第一组：多年逐月均值
    14177.572, 15055.792, 19506.788, 23359.016,
    31780.580, 41083.971, 48825.710, 41538.844,
    36095.187, 28158.081, 21198.915, 15446.667
]

group2_dict = {  # 第二组：某年逐月均值（dict，含 NaN）
    1: float("nan"), 2: 23647.567567567567, 3: 23405.962059620597, 4: 28980.414012738853,
    5: 39024.56140350877, 6: 53170.170454545456, 7: 43554.37415881561, 8: 21061.725067385443,
    9: 11958.010204081633, 10: 10852.162576687117, 11: 9593.347578347579, 12: 11107.89402173913
}
months = list(range(1, 13))
group2 = [group2_dict.get(m, float("nan")) for m in months]

# ---------- 计算差值（柱状 - 折线） ----------
diff = []
for a, b in zip(group1, group2):
    if b is None or (isinstance(b, float) and math.isnan(b)):
        diff.append(float("nan"))
    else:
        diff.append(a - b)

# ---------- 绘图 ----------
plt.figure(figsize=(11, 6))

# 1) 柱状图（第一组）
bars = plt.bar(months, group1, label="多年逐月均值（柱状）", edgecolor="black", linewidth=0.6)

# 2) 折线图（第二组）
#    对于 NaN 的月，只断开线段；markers 显示可见点
#    为了断线效果，使用 numpy.nan
g2 = np.array(group2, dtype=float)
plt.plot(months, g2, marker="o", linewidth=2, label="某年逐月均值（折线）")

# 3) 在柱顶叠加差值标注（仅当第二组不为 NaN）
for m, bar, d in zip(months, bars, diff):
    if not (isinstance(d, float) and math.isnan(d)):
        # 在柱顶略上方标注
        y = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y * 1.01,           # 稍微高于柱顶
            f"Δ={d:,.0f}",      # 千分位、无小数；需要小数可改 {d:,.1f}
            ha="center", va="bottom", fontsize=9
        )

# 轴与标题
plt.title("逐月径流量：多年均值 vs. 某年对比（含差值）", fontsize=14)
plt.xlabel("月份", fontsize=12)
plt.ylabel("径流量（单位同数据）", fontsize=12)
plt.xticks(months)

# 网格与图例
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig("/mnt/f/wsl_plot/donghai/salinity/streamflow_comparison_2022_vs_climatology.png", dpi=500)
