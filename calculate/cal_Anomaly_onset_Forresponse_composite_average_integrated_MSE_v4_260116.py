"""
Analyze Vertically Integrated MSE (VIMSE) and component anomalies for
early and late monsoon onset years.

This script:
1. Reads VIMSE and component data (vicpt, viphi, vilvq) from NetCDF files
2. Calculates climatology for March and April (1980-2021)
3. Computes composite anomalies for early and late onset years
4. Performs Student's t-test for statistical significance
5. Visualizes results with stippling for significant areas
"""

import argparse
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


EARLY_ONSET_YEARS = [1984, 1985, 1999, 2000, 2009, 2017]
LATE_ONSET_YEARS  = [1983, 1987, 1993, 1997, 2010, 2016, 2018, 2020, 2021]

CLIM_START_YEAR = 1980
CLIM_END_YEAR   = 2021

ANALYSIS_MONTHS = [3, 4]
MONTH_NAMES = {3: 'March', 4: 'April'}

VARIABLES = ['vimse', 'vicpt', 'viphi', 'vilvq']
VARIABLE_NAMES = {
    'vimse': 'VIMSE (MSE)',
    'vicpt': r'VI(c$_p$T)',
    'viphi': r'VI($\Phi$)',
    'vilvq': r'VI(L$_v$q)'
}
VARIABLE_LONG_NAMES = {
    'vimse': 'Vertically Integrated MSE',
    'vicpt': 'Vertically Integrated Sensible Heat (cp*T)',
    'viphi': 'Vertically Integrated Geopotential (Phi)',
    'vilvq': 'Vertically Integrated Latent Heat (Lv*q)'
}

# =========================
# Global plot controls
# =========================
STIPPLE_SKIP = 2                 # requirement: stippling skip=2
SCALE_FACTOR = 1e-6              # J/m^2 -> MJ/m^2

# requirement: fixed colorbar range for combined + 4x3
FIXED_VMIN_MJ = -15.0
FIXED_VMAX_MJ =  15.0


def load_all_variables(
    data_path: Path,
    file_pattern: str = "ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc",
    start_year: int = CLIM_START_YEAR,
    end_year: int = CLIM_END_YEAR,
    variables: List[str] = VARIABLES
) -> Dict[str, xr.DataArray]:
    data_dict = {var: [] for var in variables}

    for year in range(start_year, end_year + 1):
        filename = file_pattern.format(year=year)
        filepath = data_path / filename

        if not filepath.exists():
            alt_patterns = [
                f"ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc",
                f"vimse_{year}.nc",
                f"{year}_vimse.nc",
            ]
            for alt in alt_patterns:
                alt_path = data_path / alt
                if alt_path.exists():
                    filepath = alt_path
                    break
            else:
                warnings.warn(f"File not found for year {year}: {filepath}")
                continue

        try:
            ds = xr.open_dataset(filepath)
            for var in variables:
                if var in ds:
                    data_dict[var].append(ds[var])
                else:
                    warnings.warn(f"Variable '{var}' not found in {filepath}")
            ds.close()
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            continue

    result = {}
    for var in variables:
        if data_dict[var]:
            result[var] = xr.concat(data_dict[var], dim='valid_time')
        else:
            warnings.warn(f"No data found for variable '{var}'")

    return result


def load_from_single_file(filepath: Path, variables: List[str] = VARIABLES) -> Dict[str, xr.DataArray]:
    ds = xr.open_dataset(filepath)
    result = {}
    for var in variables:
        if var in ds:
            result[var] = ds[var]
        else:
            warnings.warn(f"Variable '{var}' not found in {filepath}")
    return result


def extract_month_data(data: xr.DataArray, month: int, time_dim: str = 'valid_time') -> xr.DataArray:
    time_coord = data[time_dim]
    if hasattr(time_coord.dt, 'month'):
        month_mask = time_coord.dt.month == month
    else:
        time_coord = xr.DataArray(np.array(time_coord.values, dtype='datetime64[ns]'), dims=[time_dim])
        month_mask = time_coord.dt.month == month

    data_month = data.sel({time_dim: month_mask})
    years = data_month[time_dim].dt.year.values
    data_month = data_month.assign_coords(year=(time_dim, years))
    return data_month


def calculate_climatology(
    data_month: xr.DataArray,
    start_year: int = CLIM_START_YEAR,
    end_year: int = CLIM_END_YEAR,
    time_dim: str = 'valid_time'
) -> xr.DataArray:
    years = data_month[time_dim].dt.year
    year_mask = (years >= start_year) & (years <= end_year)
    data_clim_period = data_month.sel({time_dim: year_mask})
    return data_clim_period.mean(dim=time_dim)


def calculate_composite_anomaly(
    data_month: xr.DataArray,
    climatology: xr.DataArray,
    years: List[int],
    time_dim: str = 'valid_time'
):
    year_values = data_month[time_dim].dt.year.values
    year_mask = np.isin(year_values, years)
    if not np.any(year_mask):
        raise ValueError(f"No data found for years: {years}")

    data_composite = data_month.isel({time_dim: year_mask})
    composite_mean = data_composite.mean(dim=time_dim)
    anomaly = composite_mean - climatology
    return anomaly, data_composite


def perform_ttest(
    composite_data: xr.DataArray,
    all_data: xr.DataArray,
    time_dim: str = 'valid_time',
    alpha: float = 0.10
):
    clim_mean = all_data.mean(dim=time_dim)

    lat_dim = 'latitude' if 'latitude' in composite_data.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in composite_data.dims else 'lon'

    composite_stacked = composite_data.stack(spatial=(lat_dim, lon_dim))
    clim_mean_stacked = clim_mean.stack(spatial=(lat_dim, lon_dim))

    n_composite = composite_data[time_dim].size
    composite_mean = composite_stacked.mean(dim=time_dim)
    composite_std = composite_stacked.std(dim=time_dim, ddof=1)

    t_stat = (composite_mean - clim_mean_stacked) / (composite_std / np.sqrt(n_composite))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat.values), df=n_composite - 1))

    p_value_da = xr.DataArray(p_value, coords=composite_mean.coords, dims=composite_mean.dims)

    t_stat = t_stat.unstack('spatial')
    p_value_da = p_value_da.unstack('spatial')

    return t_stat, p_value_da


def _get_lon_lat(data: xr.DataArray):
    if 'longitude' in data.coords:
        return data.longitude.values, data.latitude.values
    return data.lon.values, data.lat.values


def _subset_region(lon, lat, region: Tuple[float, float, float, float]):
    lon_mask = (lon >= region[0]) & (lon <= region[1])
    lat_mask = (lat >= region[2]) & (lat <= region[3])
    return lon_mask, lat_mask


def plot_single_variable(
    anomalies: Dict[int, Dict[str, xr.DataArray]],
    p_values: Dict[int, Dict[str, xr.DataArray]],
    output_path: Path,
    var_name: str,
    early_years: List[int],
    late_years: List[int],
    alpha: float = 0.10,
    region: Tuple[float, float, float, float] = (45, 115, -10, 30),
    cmap: str = 'RdBu_r',
    interval: float = 1.5,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    # (kept as-is; it already uses horizontal colorbar; not forced to -15..15 unless you want)
    fig, axes = plt.subplots(
        2, 2,
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )

    months = ANALYSIS_MONTHS
    onset_types = ['early', 'late']
    titles = {'early': 'Early Onset', 'late': 'Late Onset'}

    all_values = []
    for month in months:
        for onset_type in onset_types:
            d = anomalies[month][onset_type]
            flat = d.values.ravel()
            all_values.extend(flat[~np.isnan(flat)])

    vmax_data = np.percentile(np.abs(all_values), 98) * SCALE_FACTOR
    vmax_scaled = np.ceil(vmax_data / interval) * interval
    if vmax_scaled == 0:
        vmax_scaled = interval
    vmin_scaled = -vmax_scaled
    levels = np.arange(vmin_scaled, vmax_scaled + interval, interval)

    cf = None
    for i, month in enumerate(months):
        for j, onset_type in enumerate(onset_types):
            ax = axes[i, j]
            data = anomalies[month][onset_type] * SCALE_FACTOR
            pval = p_values[month][onset_type]

            lon, lat = _get_lon_lat(data)
            lon_mask, lat_mask = _subset_region(lon, lat, region)

            lon_sub = lon[lon_mask]
            lat_sub = lat[lat_mask]
            data_sub = data.values[np.ix_(lat_mask, lon_mask)]
            pval_sub = pval.values[np.ix_(lat_mask, lon_mask)]

            cf = ax.contourf(
                lon_sub, lat_sub, data_sub,
                levels=levels, cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend='both'
            )

            sig_mask = pval_sub < alpha
            lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
            ax.scatter(
                lon_mesh[::STIPPLE_SKIP, ::STIPPLE_SKIP][sig_mask[::STIPPLE_SKIP, ::STIPPLE_SKIP]],
                lat_mesh[::STIPPLE_SKIP, ::STIPPLE_SKIP][sig_mask[::STIPPLE_SKIP, ::STIPPLE_SKIP]],
                marker='.', s=0.5, c='black', alpha=0.5,
                transform=ccrs.PlateCarree()
            )

            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
            ax.set_extent(region, crs=ccrs.PlateCarree())

            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()

            ax.set_title(f"{MONTH_NAMES[month]} - {titles[onset_type]}", fontsize=11, fontweight='bold')

    fig.colorbar(
        cf, ax=axes, orientation='horizontal',
        fraction=0.05, pad=0.08, aspect=40,
        label=f'{VARIABLE_NAMES.get(var_name, var_name)} Anomaly (MJ/m²)'
    )

    confidence_pct = int((1 - alpha) * 100)
#    fig.suptitle(
#        f'{VARIABLE_LONG_NAMES.get(var_name, var_name)} Anomalies\n'
#        f'Early: {early_years} | Late: {late_years}\n'
#        f'(Stippling: {confidence_pct}% significance)',
#        fontsize=12, fontweight='bold', y=1.02
#    )

    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to: {output_path}")


def plot_early_late_separate(
    all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
    all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
    output_path: Path,
    onset_type: str,
    onset_years: List[int],
    alpha: float = 0.10,
    region: Tuple[float, float, float, float] = (45, 115, -10, 30),
    cmap: str = 'RdBu_r',
    interval: float = 1.5
) -> None:
    """
    4x3 plot with SINGLE shared HORIZONTAL colorbar.
    Colorbar range fixed to [-15, +15] MJ/m² for ALL rows/columns.
    """
    variables = VARIABLES
    columns = ['March', 'April', 'April−March']

    n_rows = len(variables)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(14, 14),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )

    # FIXED levels
    levels_ref = np.arange(FIXED_VMIN_MJ, FIXED_VMAX_MJ + interval, interval)

    cf_ref_for_cbar = None

    for row, var in enumerate(variables):
        if var not in all_anomalies:
            continue

        mar_data = all_anomalies[var][3][onset_type]
        apr_data = all_anomalies[var][4][onset_type]
        diff_data = apr_data - mar_data

        mar_pval = all_pvalues[var][3][onset_type]
        apr_pval = all_pvalues[var][4][onset_type]
        diff_pval = np.minimum(mar_pval.values, apr_pval.values)

        data_list = [mar_data, apr_data, diff_data]
        pval_list = [mar_pval.values, apr_pval.values, diff_pval]

        for col in range(n_cols):
            ax = axes[row, col]

            data = data_list[col] * SCALE_FACTOR
            pval = pval_list[col]

            lon, lat = _get_lon_lat(data)
            lon_mask, lat_mask = _subset_region(lon, lat, region)

            lon_sub = lon[lon_mask]
            lat_sub = lat[lat_mask]
            data_sub = data.values[np.ix_(lat_mask, lon_mask)]
            pval_sub = pval[np.ix_(lat_mask, lon_mask)]

            cf = ax.contourf(
                lon_sub, lat_sub, data_sub,
                levels=levels_ref,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend='both'
            )

            if row == 0 and col == n_cols - 1:
                cf_ref_for_cbar = cf

            sig_mask = pval_sub < alpha
            lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
            ax.scatter(
                lon_mesh[::STIPPLE_SKIP, ::STIPPLE_SKIP][sig_mask[::STIPPLE_SKIP, ::STIPPLE_SKIP]],
                lat_mesh[::STIPPLE_SKIP, ::STIPPLE_SKIP][sig_mask[::STIPPLE_SKIP, ::STIPPLE_SKIP]],
                marker='.', s=0.4, c='black', alpha=0.5,
                transform=ccrs.PlateCarree()
            )

            ax.coastlines(linewidth=0.4)
            ax.set_extent(region, crs=ccrs.PlateCarree())

            gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle='--')
            if col == 0:
                gl.left_labels = True
                gl.yformatter = LatitudeFormatter()
            if row == n_rows - 1:
                gl.bottom_labels = True
                gl.xformatter = LongitudeFormatter()

            if row == 0:
                ax.set_title(columns[col], fontsize=11, fontweight='bold')

            if col == 0:
                ax.text(
                    -0.15, 0.5, VARIABLE_NAMES[var],
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold',
                    va='center', ha='right',
                    rotation=90
                )

    if cf_ref_for_cbar is None:
        for ax in axes.ravel():
            if ax.get_visible() and len(ax.collections) > 0:
                cf_ref_for_cbar = ax.collections[0]
                break

    cbar = fig.colorbar(
        cf_ref_for_cbar,
        ax=axes,
        orientation='horizontal',
        fraction=0.045,
        pad=0.06,
        aspect=45,
        ticks=np.arange(FIXED_VMIN_MJ, FIXED_VMAX_MJ + 1e-9, 5.0)  # optional: ticks every 5
    )
    cbar.set_label(f'Anomaly (MJ/m²) — fixed [{FIXED_VMIN_MJ}, {FIXED_VMAX_MJ}]', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    onset_label = 'Early' if onset_type == 'early' else 'Late'
    confidence_pct = int((1 - alpha) * 100)
    fig.suptitle(
        f'{onset_label} Onset Years: {onset_years}\n'
        f'Vertically Integrated MSE and Component Anomalies\n'
        f'(Stippling: {confidence_pct}% significance)',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  {onset_label} onset figure saved to: {output_path}")


def plot_all_variables_combined(
    all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
    all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
    output_path: Path,
    early_years: List[int],
    late_years: List[int],
    alpha: float = 0.10,
    region: Tuple[float, float, float, float] = (45, 115, -10, 30),
    cmap: str = 'RdBu_r',
    interval: float = 1.5
) -> None:
    """
    4x4 combined plot with SINGLE shared HORIZONTAL colorbar.
    Colorbar range fixed to [-15, +15] MJ/m² for ALL rows/columns.
    """
    variables = VARIABLES
    months = ANALYSIS_MONTHS
    onset_types = ['early', 'late']

    n_rows = len(variables)
    n_cols = len(months) * len(onset_types)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 14),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )

    # FIXED levels
    levels_ref = np.arange(FIXED_VMIN_MJ, FIXED_VMAX_MJ + interval, interval)

    cf_ref_for_cbar = None

    for row, var in enumerate(variables):
        col = 0
        for month in months:
            for ot in onset_types:
                ax = axes[row, col]

                if var not in all_anomalies or month not in all_anomalies[var] or ot not in all_anomalies[var][month]:
                    ax.set_visible(False)
                    col += 1
                    continue

                data = all_anomalies[var][month][ot] * SCALE_FACTOR
                pval = all_pvalues[var][month][ot]

                lon, lat = _get_lon_lat(data)
                lon_mask, lat_mask = _subset_region(lon, lat, region)

                lon_sub = lon[lon_mask]
                lat_sub = lat[lat_mask]
                data_sub = data.values[np.ix_(lat_mask, lon_mask)]
                pval_sub = pval.values[np.ix_(lat_mask, lon_mask)]

                cf = ax.contourf(
                    lon_sub, lat_sub, data_sub,
                    levels=levels_ref,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )

                if row == 0 and col == n_cols - 1:
                    cf_ref_for_cbar = cf

                sig_mask = pval_sub < alpha
                lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
                ax.scatter(
                    lon_mesh[::STIPPLE_SKIP, ::STIPPLE_SKIP][sig_mask[::STIPPLE_SKIP, ::STIPPLE_SKIP]],
                    lat_mesh[::STIPPLE_SKIP, ::STIPPLE_SKIP][sig_mask[::STIPPLE_SKIP, ::STIPPLE_SKIP]],
                    marker='.', s=0.3, c='black', alpha=0.5,
                    transform=ccrs.PlateCarree()
                )

                ax.coastlines(linewidth=0.4)
                ax.set_extent(region, crs=ccrs.PlateCarree())

                gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle='--')
                if col == 0:
                    gl.left_labels = True
                    gl.yformatter = LatitudeFormatter()
                if row == n_rows - 1:
                    gl.bottom_labels = True
                    gl.xformatter = LongitudeFormatter()

                if row == 0:
                    onset_label = 'Early' if ot == 'early' else 'Late'
                    ax.set_title(f"{MONTH_NAMES[month]}\n{onset_label}", fontsize=10, fontweight='bold')

                if col == 0:
                    ax.text(
                        -0.15, 0.5, VARIABLE_NAMES[var],
                        transform=ax.transAxes,
                        fontsize=10, fontweight='bold',
                        va='center', ha='right',
                        rotation=90
                    )

                col += 1

    if cf_ref_for_cbar is None:
        for ax in axes.ravel():
            if ax.get_visible() and len(ax.collections) > 0:
                cf_ref_for_cbar = ax.collections[0]
                break

    cbar = fig.colorbar(
        cf_ref_for_cbar,
        ax=axes,
        orientation='horizontal',
        fraction=0.045,
        pad=0.06,
        aspect=55,
        ticks=np.arange(FIXED_VMIN_MJ, FIXED_VMAX_MJ + 1e-9, 5.0)
    )
    cbar.set_label(f'Anomaly (MJ/m²) — fixed [{FIXED_VMIN_MJ}, {FIXED_VMAX_MJ}]', fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    confidence_pct = int((1 - alpha) * 100)
    fig.suptitle(
        'Vertically Integrated MSE and Component Anomalies\n'
        f'Early Onset: {early_years} | Late Onset: {late_years}\n'
        f'(Stippling: {confidence_pct}% significance)',
        fontsize=13, fontweight='bold', y=1.01
    )

    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined figure saved to: {output_path}")


def save_results_netcdf(
    all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
    all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
    all_climatologies: Dict[str, Dict[int, xr.DataArray]],
    output_path: Path,
    early_years: List[int],
    late_years: List[int]
) -> None:
    ds = xr.Dataset()

    for var in VARIABLES:
        if var not in all_anomalies:
            continue

        for month in ANALYSIS_MONTHS:
            if month not in all_anomalies[var]:
                continue

            month_name = MONTH_NAMES[month].lower()

            if var in all_climatologies and month in all_climatologies[var]:
                ds[f'{var}_clim_{month_name}'] = all_climatologies[var][month]
                ds[f'{var}_clim_{month_name}'].attrs = {
                    'long_name': f'{VARIABLE_LONG_NAMES[var]} Climatology for {MONTH_NAMES[month]}',
                    'units': 'J m-2',
                    'period': f'{CLIM_START_YEAR}-{CLIM_END_YEAR}'
                }

            for onset_type in ['early', 'late']:
                years_list = early_years if onset_type == 'early' else late_years

                ds[f'{var}_anom_{month_name}_{onset_type}'] = all_anomalies[var][month][onset_type]
                ds[f'{var}_anom_{month_name}_{onset_type}'].attrs = {
                    'long_name': f'{VARIABLE_LONG_NAMES[var]} Anomaly for {MONTH_NAMES[month]} ({onset_type} onset)',
                    'units': 'J m-2',
                    'years': str(years_list)
                }

                ds[f'{var}_pvalue_{month_name}_{onset_type}'] = all_pvalues[var][month][onset_type]
                ds[f'{var}_pvalue_{month_name}_{onset_type}'].attrs = {
                    'long_name': f'T-test p-value for {VARIABLE_LONG_NAMES[var]} {MONTH_NAMES[month]} ({onset_type} onset)',
                    'units': '1',
                    'test': 'one-sample t-test against climatology'
                }

    ds.attrs = {
        'title': 'VIMSE and Component Anomaly Analysis for Early/Late Monsoon Onset',
        'early_onset_years': str(early_years),
        'late_onset_years': str(late_years),
        'climatology_period': f'{CLIM_START_YEAR}-{CLIM_END_YEAR}',
        'variables': ', '.join(VARIABLES),
        'Conventions': 'CF-1.7'
    }

    ds.to_netcdf(output_path)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze VIMSE and component anomalies for early/late monsoon onset years.',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('input', type=str,
                        help='Input path: directory with yearly VIMSE files or single combined file')
    parser.add_argument('--single-file', action='store_true',
                        help='Input is a single combined NetCDF file')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for PDF plots (default: current directory)')
    parser.add_argument('--output-nc', type=str, default=None,
                        help='Output NetCDF path for results (optional)')
    parser.add_argument('--file-pattern', type=str,
                        default='ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc',
                        help='File pattern with {year} placeholder (for directory input)')
    parser.add_argument('--region', type=float, nargs=4,
                        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
                        default=[45, 115, -10, 30],
                        help='Plot region extent (default: 45 115 -10 30)')
    parser.add_argument('--early-years', type=str, default=None,
                        help='Comma-separated list of early onset years')
    parser.add_argument('--late-years', type=str, default=None,
                        help='Comma-separated list of late onset years')
    parser.add_argument('--alpha', type=float, default=0.10,
                        help='Significance level for t-test (default: 0.10 for 90%% confidence)')
    parser.add_argument('--interval', type=float, default=1.5,
                        help='Colorbar interval in MJ/m² (default: 1.5)')
    parser.add_argument('--no-individual', action='store_true',
                        help='Skip individual variable plots, only create combined plot')

    args = parser.parse_args()

    early_years = [int(y.strip()) for y in args.early_years.split(',')] if args.early_years else list(EARLY_ONSET_YEARS)
    late_years  = [int(y.strip()) for y in args.late_years.split(',')] if args.late_years else list(LATE_ONSET_YEARS)

    print("=" * 70)
    print("VIMSE and Component Anomaly Analysis for Monsoon Onset")
    print("=" * 70)
    print(f"Early onset years: {early_years}")
    print(f"Late onset years: {late_years}")
    print(f"Climatology period: {CLIM_START_YEAR}-{CLIM_END_YEAR}")
    print(f"Analysis months: {[MONTH_NAMES[m] for m in ANALYSIS_MONTHS]}")
    print(f"Significance level: {args.alpha} ({int((1-args.alpha)*100)}% confidence)")
    print(f"Variables: {VARIABLES}")
    print(f"Stippling skip: {STIPPLE_SKIP}")
    print(f"Fixed colorbar range (combined + 4x3): [{FIXED_VMIN_MJ}, {FIXED_VMAX_MJ}] MJ/m²")
    print()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")

    if args.single_file:
        data_dict = load_from_single_file(input_path, VARIABLES)
    else:
        data_dict = load_all_variables(input_path, args.file_pattern,
                                       CLIM_START_YEAR, CLIM_END_YEAR, VARIABLES)

    print(f"Loaded variables: {list(data_dict.keys())}")

    sample_var = list(data_dict.values())[0]
    time_dim = 'valid_time' if 'valid_time' in sample_var.dims else 'time'
    print(f"Time dimension: {time_dim}")
    print()

    all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]] = {}
    all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]] = {}
    all_climatologies: Dict[str, Dict[int, xr.DataArray]] = {}

    for var in VARIABLES:
        if var not in data_dict:
            print(f"Skipping {var} - not found in data")
            continue

        print(f"Processing {VARIABLE_LONG_NAMES[var]}...")
        data = data_dict[var]

        all_anomalies[var] = {}
        all_pvalues[var] = {}
        all_climatologies[var] = {}

        for month in ANALYSIS_MONTHS:
            print(f"  {MONTH_NAMES[month]}...")

            data_month = extract_month_data(data, month, time_dim)

            climatology = calculate_climatology(data_month, CLIM_START_YEAR, CLIM_END_YEAR, time_dim)
            all_climatologies[var][month] = climatology

            all_anomalies[var][month] = {}
            all_pvalues[var][month] = {}

            for onset_type, years in [('early', early_years), ('late', late_years)]:
                anomaly, composite_data = calculate_composite_anomaly(
                    data_month, climatology, years, time_dim
                )
                all_anomalies[var][month][onset_type] = anomaly

                _, pval = perform_ttest(composite_data, data_month, time_dim, args.alpha)
                all_pvalues[var][month][onset_type] = pval

                n_sig = (pval < args.alpha).sum().values
                total = pval.size
                print(f"    {onset_type.capitalize()}: {n_sig}/{total} significant ({100*n_sig/total:.1f}%)")

    print()

    region = tuple(args.region)

    if not args.no_individual:
        print("Creating individual variable plots...")
        for var in VARIABLES:
            if var not in all_anomalies:
                continue
            output_path = output_dir / f'{var}_anomaly_mar_apr.pdf'
            plot_single_variable(
                all_anomalies[var], all_pvalues[var],
                output_path, var,
                early_years, late_years,
                alpha=args.alpha,
                region=region,
                interval=args.interval
            )

    print("Creating combined plot...")
    combined_output = output_dir / 'vimse_components_anomaly_combined.pdf'
    plot_all_variables_combined(
        all_anomalies, all_pvalues,
        combined_output,
        early_years, late_years,
        alpha=args.alpha,
        region=region,
        interval=args.interval
    )

    print("Creating 4x3 panel plots (March / April / April-March)...")

    early_output = output_dir / 'vimse_components_early_onset_4x3.pdf'
    plot_early_late_separate(
        all_anomalies, all_pvalues,
        early_output,
        onset_type='early',
        onset_years=early_years,
        alpha=args.alpha,
        region=region,
        interval=args.interval
    )

    late_output = output_dir / 'vimse_components_late_onset_4x3.pdf'
    plot_early_late_separate(
        all_anomalies, all_pvalues,
        late_output,
        onset_type='late',
        onset_years=late_years,
        alpha=args.alpha,
        region=region,
        interval=args.interval
    )

    if args.output_nc:
        print("Saving results to NetCDF...")
        save_results_netcdf(
            all_anomalies, all_pvalues, all_climatologies,
            Path(args.output_nc), early_years, late_years
        )

    print()
    print("=" * 70)
    print("Analysis complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
