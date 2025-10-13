# -*- coding: utf-8 -*-
"""
Overlay two monthly streamflow series:
1) Group 1 (climatology) as a semi-transparent bar chart
2) Group 2 (e.g., 2022) as an orange-red line on top of bars
3) Hatched "difference bars" showing |Group2 - Group1|, starting at min(Group1, Group2),
   with clear styling and in-bar labels: signed diff and % vs. climatology
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- Data ----------
group1 = [  # Climatological monthly means
    14177.572, 15055.792, 19506.788, 23359.016,
    31780.580, 41083.971, 48825.710, 41538.844,
    36095.187, 28158.081, 21198.915, 15446.667
]

group2_dict = {  # Target year monthly means (dict, may contain NaN)
    1: float("nan"), 2: 23647.567567567567, 3: 23405.962059620597, 4: 28980.414012738853,
    5: 39024.56140350877, 6: 53170.170454545456, 7: 43554.37415881561, 8: 21061.725067385443,
    9: 11958.010204081633, 10: 10852.162576687117, 11: 9593.347578347579, 12: 11107.89402173913
}
months = list(range(1, 13))
group2 = [group2_dict.get(m, float("nan")) for m in months]

g1 = np.array(group1, dtype=float)
g2 = np.array(group2, dtype=float)

# Differences
diff_abs = np.abs(g1 - g2)      # height for hatched difference bars
diff_signed = g2 - g1           # signed difference for labels (year - clim)

# Bottom & height for difference bars (skip when g2 is NaN)
bottom = np.where(np.isnan(g2), np.nan, np.minimum(g1, g2))
height = np.where(np.isnan(g2), 0.0, diff_abs)

# ---------- Plot ----------
plt.figure(figsize=(11, 6))

# 1) Bars: climatology (more transparent)
bars = plt.bar(
    months, g1,
    label="Climatology (bars)",
    edgecolor="black",
    linewidth=0.6,
    alpha=0.35,       # more transparent than before
    zorder=1
)

# 2) Hatched difference bars: make them more prominent
#    - denser hatch
#    - orange-red edgecolor
#    - light orange-red facecolor with some transparency
diff_bars = plt.bar(
    months, height,
    bottom=bottom,
    width=0.8,
    label="Difference band (|Year - Climatology|)",
    facecolor=(1.0, 0.4, 0.0, 0.22),  # light orange-red with transparency
    edgecolor="navajowhite",      # orange-red edge
    hatch="///",                     # denser hatching
    linewidth=0.65,
    zorder=2.6
)

# 3) Line: target year (orange-red), on top of bars
plt.plot(
    months, g2,
    marker="o",
    linewidth=2.4,
    markersize=5.5,
    label="2022 (line)",
    color="orangered",
    zorder=3.5
)

# 4) Labels centered in the difference bars: ±diff and % vs. climatology
for m, a, b, h, bot, ds in zip(months, g1, g2, height, bottom, diff_signed):
    if np.isnan(b) or h <= 0:
        continue
    y_mid = float(bot) + float(h) / 2.0
    if a and not np.isnan(a):
        pct = ds / a * 100.0
        label = f"{ds:+,.0f} ({pct:+.1f}%)"
    else:
        label = f"{ds:+,.0f}"
    txt_color = "orangered" if ds >= 0 else "steelblue"
    plt.text(
        m, y_mid, label,
        ha="center", va="center", fontsize=9, color=txt_color,
        zorder=4,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.65, linewidth=0)
    )

# Axes & title (English)
plt.title("Monthly Runoff: Climatology vs. Year (with Difference Bands)", fontsize=14)
plt.xlabel("Month", fontsize=12)
plt.ylabel("Runoff", fontsize=12)
plt.xticks(months)

# Grid & legend
plt.grid(axis="y", linestyle="--", alpha=0.6, zorder=0)
plt.legend()

plt.tight_layout()
# Save or show
# plt.show()
plt.savefig("/mnt/f/wsl_plot/donghai/salinity/streamflow_comparison_2022_vs_climatology.png", dpi=500)
