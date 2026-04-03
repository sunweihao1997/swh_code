"""
2026-01-08 (Updated)
Count annual TS+ storm numbers in the Northwest Pacific (WP) WITHOUT marine sub-regions.

Definition:
- Each storm (unique SID) is counted ONCE, assigned to the year of its first record (genesis year).
- Only storms that ever reach max wind >= 34 kt (Tropical Storm threshold) are counted.
"""

import pandas as pd
import matplotlib.pyplot as plt


def count_tsplus_wp_annual(
    ibtracs_path: str,
    output_csv_path: str,
    output_fig_path: str,
    basin_filter: str = "WP",
    intensity_threshold_kt: float = 34.0,  # TS threshold
):
    print("Reading IBTrACS CSV...")
    df = pd.read_csv(ibtracs_path, low_memory=False, skiprows=[1])

    # Required columns
    for c in ["SID", "ISO_TIME"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}. Available columns: {list(df.columns)}")

    # Candidate wind columns (knots). Preference order can be adjusted.
    wind_candidates = [
        "USA_WIND",  # often JTWC 1-min winds for WP
        "WMO_WIND",  # WMO; may be missing or sparse depending on record
        "CMA_WIND",
        "JMA_WIND",
        "HKO_WIND",
        "TOKYO_WIND",
        "REUNION_WIND",
        "NEWDELHI_WIND",
        "BOM_WIND",
    ]
    wind_col = next((c for c in wind_candidates if c in df.columns), None)
    if wind_col is None:
        raise ValueError(
            "No usable wind column found. Tried: "
            + ", ".join(wind_candidates)
            + f". Available columns: {list(df.columns)}"
        )

    use_cols = ["SID", "ISO_TIME", wind_col] + (["BASIN"] if "BASIN" in df.columns else [])
    df = df[use_cols].copy()

    # Optional basin filter (robust if you feed a global IBTrACS file)
    if "BASIN" in df.columns and basin_filter:
        df = df[df["BASIN"].astype(str).str.upper().eq(basin_filter.upper())].copy()

    # Parse time
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    df = df.dropna(subset=["ISO_TIME", "SID"])

    # TS+ filter by max wind over lifetime
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
    max_wind = df.groupby("SID")[wind_col].max()
    keep_sids = max_wind[max_wind >= intensity_threshold_kt].index
    df = df[df["SID"].isin(keep_sids)].copy()

    print(
        f"TS+ filter: kept {len(keep_sids)} storms with max {wind_col} >= {intensity_threshold_kt} kt "
        f"in basin {basin_filter}."
    )

    # Assign each SID to genesis year (first appearance)
    genesis_time = df.groupby("SID")["ISO_TIME"].min()
    genesis_year = genesis_time.dt.year

    annual_counts = genesis_year.value_counts().sort_index()
    annual_counts.index.name = "Year"
    annual_counts.name = "Storm_Count_TSplus"

    # Fill missing years with 0 for continuity
    year_min, year_max = int(annual_counts.index.min()), int(annual_counts.index.max())
    full_years = pd.Index(range(year_min, year_max + 1), name="Year")
    annual_counts = annual_counts.reindex(full_years, fill_value=0)

    result = annual_counts.reset_index()
    result.to_csv(output_csv_path, index=False)
    print(f"Saved CSV: {output_csv_path}")

    # Climatology (annual mean)
    climatology = annual_counts.mean()
    print(f"Climatology (annual mean, {year_min}-{year_max}): {climatology:.2f}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(result["Year"], result["Storm_Count_TSplus"])
    ax.axhline(climatology, linestyle="--", linewidth=2, label=f"Climatology mean = {climatology:.2f}")

    run_mean = annual_counts.rolling(window=5, center=True, min_periods=1).mean()
    ax.plot(result["Year"], run_mean.values, linewidth=2, label="5-yr running mean")

    ax.set_title(
        f"Annual TS+ Storm Count in Northwest Pacific ({basin_filter})\n"
        f"({year_min}-{year_max}, counted by genesis year; threshold >= {intensity_threshold_kt} kt using {wind_col})"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Count (unique SID)")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_fig_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {output_fig_path}")


def main():
    ibtracs_csv_path = "/home/sun/data/download_data/IBTRACS_typhoon/ibtracs.WP.list.v04r01.csv"

    output_csv_path = "/home/sun/data/process/analysis/typhoon_prediction/wp_tsplus_count.csv"
    output_fig_path = "/home/sun/data/process/analysis/typhoon_prediction/wp_tsplus_count.png"

    count_tsplus_wp_annual(
        ibtracs_path=ibtracs_csv_path,
        output_csv_path=output_csv_path,
        output_fig_path=output_fig_path,
        basin_filter="WP",
        intensity_threshold_kt=34.0,  # TS and above
    )


if __name__ == "__main__":
    main()
