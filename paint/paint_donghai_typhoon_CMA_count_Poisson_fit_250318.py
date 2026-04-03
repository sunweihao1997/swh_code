from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson

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

grade_order = [2, 3, 4, 5, 6]
df = df[df["max_grade_code"].isin(grade_order)].copy()

# =========================
# 3. Count annual total
# =========================
year_total = df.groupby("year").size()

all_years = list(range(1960, int(df["year"].max()) + 1))
year_total = year_total.reindex(all_years, fill_value=0)

# annual count series
y = year_total.values

# =========================
# 4. Poisson fit
# =========================
lam = y.mean()
var_y = y.var(ddof=1)
vmr = var_y / lam if lam > 0 else np.nan   # variance-to-mean ratio

k = np.arange(y.min(), y.max() + 1)
pmf = poisson.pmf(k, mu=lam)

# =========================
# 5. Plot empirical histogram + Poisson fit
# =========================
fig, ax = plt.subplots(figsize=(10, 6))

# histogram bins centered on integers
bins = np.arange(y.min() - 0.5, y.max() + 1.5, 1)

ax.hist(
    y,
    bins=bins,
    density=True,
    alpha=0.65,
    color="#AFC4D8",
    edgecolor="white",
    linewidth=0.8,
    label="Empirical histogram",
)

ax.plot(
    k,
    pmf,
    "o-",
    color="#A63232",
    linewidth=2.2,
    markersize=5,
    label=f"Poisson fit ($\\lambda$={lam:.2f})",
)

# labels and title
ax.set_title("Distribution of Annual Typhoon Counts Since 1960 (TS and Above)", fontsize=15, pad=12)
ax.set_xlabel("Annual typhoon count", fontsize=12)
ax.set_ylabel("Probability", fontsize=12)

ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# annotation
text_str = (
    f"Years: {len(y)}\n"
    f"Mean = {lam:.2f}\n"
    f"Variance = {var_y:.2f}\n"
    f"Var/Mean = {vmr:.2f}"
)
ax.text(
    0.98, 0.95, text_str,
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=10,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor="0.8")
)

ax.legend(frameon=False)

plt.tight_layout()

out_png = out_dir / "annual_typhoon_count_histogram_poisson_fit_since_1960.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {out_png}")