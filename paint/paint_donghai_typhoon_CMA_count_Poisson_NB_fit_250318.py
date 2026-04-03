from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson, nbinom

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

y = year_total.values

# =========================
# 4. Fit Poisson and Negative Binomial
# =========================
mu = y.mean()
var_y = y.var(ddof=1)
vmr = var_y / mu if mu > 0 else np.nan

# x range for pmf
k = np.arange(y.min(), y.max() + 1)

# Poisson fit
poisson_pmf = poisson.pmf(k, mu=mu)

# Negative Binomial fit by method of moments
# scipy.stats.nbinom parameterization:
# mean = n * (1-p) / p
# var  = n * (1-p) / p^2
# => n = mu^2 / (var - mu), p = n / (n + mu)
nbinom_available = var_y > mu

if nbinom_available:
    r = mu**2 / (var_y - mu)
    p = r / (r + mu)
    nbinom_pmf = nbinom.pmf(k, n=r, p=p)
else:
    r = np.nan
    p = np.nan
    nbinom_pmf = None

# =========================
# 5. Plot
# =========================
fig, ax = plt.subplots(figsize=(11, 6.5))

# Empirical histogram
bins = np.arange(y.min() - 0.5, y.max() + 1.5, 1)
ax.hist(
    y,
    bins=bins,
    density=True,
    alpha=0.60,
    color="#AFC4D8",
    edgecolor="white",
    linewidth=0.8,
    label="Empirical histogram",
)

# Poisson fit
ax.plot(
    k,
    poisson_pmf,
    "o-",
    color="#A63232",
    linewidth=2.4,
    markersize=5.5,
    label=f"Poisson fit ($\\lambda$={mu:.2f})",
)

# Negative Binomial fit
if nbinom_available:
    ax.plot(
        k,
        nbinom_pmf,
        "s--",
        color="#2E6F95",
        linewidth=2.2,
        markersize=5,
        label=f"Negative binomial fit (r={r:.2f}, p={p:.3f})",
    )

# Labels
ax.set_title(
    "Distribution of Annual Typhoon Counts Since 1960 (TS and Above)",
    fontsize=16,
    pad=12,
)
ax.set_xlabel("Annual typhoon count", fontsize=13)
ax.set_ylabel("Probability", fontsize=13)

# Style
ax.grid(axis="y", linestyle="--", alpha=0.25)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Text box
text_str = (
    f"Years: {len(y)}\n"
    f"Mean = {mu:.2f}\n"
    f"Variance = {var_y:.2f}\n"
    f"Var/Mean = {vmr:.2f}"
)
if nbinom_available:
    text_str += f"\nNB r = {r:.2f}\nNB p = {p:.3f}"
else:
    text_str += "\nNB fit unavailable\n(variance <= mean)"

ax.text(
    0.98, 0.95, text_str,
    transform=ax.transAxes,
    ha="right", va="top",
    fontsize=10.5,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        alpha=0.88,
        edgecolor="0.75"
    )
)

ax.legend(frameon=False, fontsize=11)

plt.tight_layout()

out_png = out_dir / "annual_typhoon_count_histogram_poisson_nbinom_fit_since_1960.png"
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.show()

print(f"Figure saved to: {out_png}")