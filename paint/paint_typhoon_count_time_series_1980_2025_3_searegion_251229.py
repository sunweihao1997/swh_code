import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path

# =============================
# User config
# =============================
csv_path = Path("/home/sun/wd_14/data/data/process/analysis/typhoon_prediction/typhoon_count.csv")  # 改成你的CSV路径
regions_wanted = ["Eastern China Sea", "South China Sea", "Yellow Sea"]
start_year, end_year = 1980, 2025
out_png = Path("/home/sun/paint/donghai/typhoon_prediction/typhoon_counts_timeseries_1980_2025_scattermarkers.png")

# =============================
# Load
# =============================
df = pd.read_csv(csv_path)

def pick_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None

year_col   = pick_col(df.columns, ["Year", "year", "YEAR"])
region_col = pick_col(df.columns, ["Region", "region", "REGION"])
count_col  = pick_col(df.columns, ["Typhoon_Count", "typhoon_count", "count", "Count", "TY_COUNT", "TC_COUNT"])

if not (year_col and region_col and count_col):
    raise ValueError(f"CSV列名不匹配。当前列：{list(df.columns)}")

df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
df[count_col] = pd.to_numeric(df[count_col], errors="coerce")

subset = df[df[year_col].between(start_year, end_year) & df[region_col].isin(regions_wanted)].copy()

# 透视成长表 -> 宽表（Year x Region）
years = list(range(start_year, end_year + 1))
wide = (
    subset.pivot_table(index=year_col, columns=region_col, values=count_col, aggfunc="sum")
    .reindex(years)
    .fillna(0)
)

# 防止某个海区缺列
for r in regions_wanted:
    if r not in wide.columns:
        wide[r] = 0
wide = wide[regions_wanted].astype(int)

# =============================
# Plot: line + different scatter markers
# =============================
markers = {
    "Eastern China Sea": "o",  # circle
    "South China Sea": "s",    # square
    "Yellow Sea": "^",         # triangle up
}

plt.figure(figsize=(12, 6))

# 先画线（不带marker）
for r in regions_wanted:
    plt.plot(wide.index, wide[r], label=r, linewidth=2.2)

# 再叠加 scatter（每个区域不同marker）
for r in regions_wanted:
    plt.scatter(
        wide.index,
        wide[r],
        marker=markers.get(r, "o"),
        s=18,
        alpha=0.9,
    )

plt.title("Typhoon Counts by Region (1980–2025)")
plt.xlabel("Year")
plt.ylabel("Typhoon Count")

ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
plt.xticks(range(start_year, end_year + 1, 5), rotation=0)

plt.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35)
plt.legend(title="Region", frameon=False, loc="upper left")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {out_png.resolve()}")
