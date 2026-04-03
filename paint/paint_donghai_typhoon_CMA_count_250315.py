from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# =========================
# 0. Basic style
# =========================
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1. Read data
# =========================
csv_path = Path("/home/sun/data/download_data/CMA_typhoon/CMABSTdata_output/all_storms.csv")
out_dir = csv_path.parent

df = pd.read_csv(csv_path)

# =========================
# 2. Filter data
#    Since 1960, TS and above only
#    max_grade_code:
#    2=TS, 3=STS, 4=TY, 5=STY, 6=SuperTY
# =========================
df = df[df["year"] >= 1960].copy()
df = df[df["max_grade_code"] >= 2].copy()

grade_map = {
    2: "TS",
    3: "STS",
    4: "TY",
    5: "STY",
    6: "SuperTY",
}
grade_order = [2, 3, 4, 5, 6]

df = df[df["max_grade_code"].isin(grade_order)].copy()
df["grade_name"] = df["max_grade_code"].map(grade_map)

# =========================
# 3. Count by year × grade
# =========================
year_grade_count = (
    df.groupby(["year", "max_grade_code"])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=grade_order, fill_value=0)
)

all_years = list(range(1960, int(df["year"].max()) + 1))
year_grade_count = year_grade_count.reindex(all_years, fill_value=0)
year_grade_count.columns = [grade_map[g] for g in grade_order]

# Annual total
year_total = year_grade_count.sum(axis=1)

# Ratio of SuperTY in annual total
superty_ratio = (
    year_grade_count["SuperTY"]
    .div(year_total.replace(0, pd.NA))
    .fillna(0)
    * 100
)

# =========================
# 4. Lighter palette + dark red line
# =========================
colors = {
    "TS": "#AFC8BE",       # medium sage
    "STS": "#D1B79C",      # muted sand
    "TY": "#AFC4D8",       # muted blue
    "STY": "#C5B0CD",      # muted lavender
    "SuperTY": "#E3A999",  # muted salmon
}
ratio_line_color = "#A63232"   # dark red

# =========================
# 5. Plot stacked bars + ratio line
# =========================
fig, ax1 = plt.subplots(figsize=(18, 8))

bottom = None
bar_handles = []
bar_labels = []

for col in year_grade_count.columns:
    bars = ax1.bar(
        year_grade_count.index,
        year_grade_count[col],
        bottom=bottom,
        label=col,
        color=colors[col],
        width=0.82,
        edgecolor="white",
        linewidth=0.5,
    )
    bar_handles.append(bars)
    bar_labels.append(col)

    if bottom is None:
        bottom = year_grade_count[col].values.copy()
    else:
        bottom = bottom + year_grade_count[col].values

# Left axis
ax1.set_title("Annual Typhoon Counts Since 1960 (TS and Above)", fontsize=16, pad=14)
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Count", fontsize=12)
ax1.grid(axis="y", linestyle="--", alpha=0.22)

# Clean spines
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# x-axis ticks every 5 years
years = year_grade_count.index.tolist()
xticks = years[::5]
ax1.set_xticks(xticks)
ax1.set_xticklabels(xticks, rotation=45)

# Annotate annual totals
for x, total in zip(year_grade_count.index, year_total):
    if total > 0:
        ax1.text(x, total + 0.15, f"{int(total)}", ha="center", va="bottom", fontsize=8)

# Right axis for SuperTY ratio
ax2 = ax1.twinx()
line_ratio, = ax2.plot(
    year_grade_count.index,
    superty_ratio,
    color=ratio_line_color,
    linewidth=2.6,
    linestyle="-",
    marker="o",
    markersize=4.5,
    label="SuperTY Ratio",
)
ax2.set_ylabel("Ratio (%)", fontsize=12, color=ratio_line_color)
ax2.tick_params(axis="y", colors=ratio_line_color)
ax2.set_ylim(0, 70)
ax2.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
ax2.spines["top"].set_visible(False)

# Combined legend
handles = [h[0] for h in bar_handles] + [line_ratio]
labels = bar_labels + ["SuperTY Ratio"]
ax1.legend(handles, labels, title="Category", ncol=6, frameon=False, loc="upper left")

plt.tight_layout()

out_png = out_dir / "annual_typhoon_counts_with_superty_ratio_since_1960.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {out_png}")