"""
260115
Analyze Vertically Integrated MSE (VIMSE) anomalies for early and late monsoon onset years.

This script:
1. Reads VIMSE data from NetCDF files (computed by compute_mse.py)
2. Calculates climatology for April and May (1980-2021)
3. Computes composite anomalies for early and late onset years
4. Performs Student's t-test for statistical significance
5. Visualizes results with stippling for significant areas

Author: Generated script for ERA5 VIMSE analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import warnings

import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# Define early and late onset years based on the provided onset date information
EARLY_ONSET_YEARS = [1984, 1985, 1999, 2000, 2009, 2017]
LATE_ONSET_YEARS = [1983, 1987, 1993, 1997, 2010, 2016, 2018, 2020, 2021]

# Analysis period
CLIM_START_YEAR = 1980
CLIM_END_YEAR = 2021

# Months to analyze (April=4, May=5)
ANALYSIS_MONTHS = [4, 5]
MONTH_NAMES = {4: 'April', 5: 'May'}


def load_vimse_data(data_path: Path, 
                    file_pattern: str = "ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc",
                    start_year: int = CLIM_START_YEAR,
                    end_year: int = CLIM_END_YEAR,
                    vimse_var: str = 'vimse') -> xr.DataArray:
    """
    Load VIMSE data from multiple yearly files.
    
    Parameters
    ----------
    data_path : Path
        Directory containing VIMSE NetCDF files
    file_pattern : str
        Filename pattern with {year} placeholder
    start_year : int
        Start year of analysis period
    end_year : int
        End year of analysis period
    vimse_var : str
        Name of VIMSE variable in files
        
    Returns
    -------
    xr.DataArray
        Combined VIMSE data with time dimension
    """
    datasets = []
    
    for year in range(start_year, end_year + 1):
        filename = file_pattern.format(year=year)
        filepath = data_path / filename
        
        if not filepath.exists():
            # Try alternative patterns
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
            if vimse_var in ds:
                datasets.append(ds[vimse_var])
            else:
                # Try to find the variable
                data_vars = list(ds.data_vars)
                if len(data_vars) == 1:
                    datasets.append(ds[data_vars[0]])
                else:
                    warnings.warn(f"Variable '{vimse_var}' not found in {filepath}. "
                                f"Available: {data_vars}")
            ds.close()
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            continue
    
    if not datasets:
        raise FileNotFoundError(f"No VIMSE data files found in {data_path}")
    
    # Concatenate along time dimension
    vimse = xr.concat(datasets, dim='valid_time')
    
    return vimse


def load_vimse_from_single_file(filepath: Path, vimse_var: str = 'vimse') -> xr.DataArray:
    """
    Load VIMSE data from a single combined NetCDF file.
    
    Parameters
    ----------
    filepath : Path
        Path to combined VIMSE NetCDF file
    vimse_var : str
        Name of VIMSE variable
        
    Returns
    -------
    xr.DataArray
        VIMSE data
    """
    ds = xr.open_dataset(filepath)
    
    if vimse_var in ds:
        vimse = ds[vimse_var]
    else:
        data_vars = list(ds.data_vars)
        if len(data_vars) == 1:
            vimse = ds[data_vars[0]]
        else:
            raise KeyError(f"Variable '{vimse_var}' not found. Available: {data_vars}")
    
    return vimse


def extract_month_data(vimse: xr.DataArray, 
                       month: int,
                       time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Extract data for a specific month across all years.
    
    Parameters
    ----------
    vimse : xr.DataArray
        VIMSE data with time dimension
    month : int
        Month number (1-12)
    time_dim : str
        Name of time dimension
        
    Returns
    -------
    xr.DataArray
        Data for the specified month with year as dimension
    """
    # Get time coordinate
    time_coord = vimse[time_dim]
    
    # Extract month
    if hasattr(time_coord.dt, 'month'):
        month_mask = time_coord.dt.month == month
    else:
        # Convert to datetime if needed
        time_coord = xr.DataArray(
            np.array(time_coord.values, dtype='datetime64[ns]'),
            dims=[time_dim]
        )
        month_mask = time_coord.dt.month == month
    
    vimse_month = vimse.sel({time_dim: month_mask})
    
    # Add year coordinate for easier selection
    years = vimse_month[time_dim].dt.year.values
    vimse_month = vimse_month.assign_coords(year=(time_dim, years))
    
    return vimse_month


def calculate_climatology(vimse_month: xr.DataArray,
                          start_year: int = CLIM_START_YEAR,
                          end_year: int = CLIM_END_YEAR,
                          time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Calculate climatological mean for a specific month.
    
    Parameters
    ----------
    vimse_month : xr.DataArray
        VIMSE data for a specific month
    start_year : int
        Start year for climatology
    end_year : int
        End year for climatology
    time_dim : str
        Name of time dimension
        
    Returns
    -------
    xr.DataArray
        Climatological mean
    """
    # Filter years
    years = vimse_month[time_dim].dt.year
    year_mask = (years >= start_year) & (years <= end_year)
    vimse_clim_period = vimse_month.sel({time_dim: year_mask})
    
    # Calculate mean
    climatology = vimse_clim_period.mean(dim=time_dim)
    
    return climatology


def calculate_composite_anomaly(vimse_month: xr.DataArray,
                                climatology: xr.DataArray,
                                years: List[int],
                                time_dim: str = 'valid_time') -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate composite mean anomaly for selected years.
    
    Parameters
    ----------
    vimse_month : xr.DataArray
        VIMSE data for a specific month
    climatology : xr.DataArray
        Climatological mean
    years : list
        List of years to composite
    time_dim : str
        Name of time dimension
        
    Returns
    -------
    tuple
        (composite_anomaly, composite_data) - anomaly and raw composite values
    """
    # Select years for composite
    year_values = vimse_month[time_dim].dt.year.values
    year_mask = np.isin(year_values, years)
    
    if not np.any(year_mask):
        raise ValueError(f"No data found for years: {years}")
    
    vimse_composite = vimse_month.isel({time_dim: year_mask})
    
    # Calculate composite mean
    composite_mean = vimse_composite.mean(dim=time_dim)
    
    # Calculate anomaly
    anomaly = composite_mean - climatology
    
    return anomaly, vimse_composite


def perform_ttest(composite_data: xr.DataArray,
                  all_data: xr.DataArray,
                  time_dim: str = 'valid_time',
                  alpha: float = 0.05) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Perform Student's t-test comparing composite years against climatology.
    
    Uses a one-sample t-test: tests if the mean of composite years 
    significantly differs from the climatological mean.
    
    Parameters
    ----------
    composite_data : xr.DataArray
        Data for composite years
    all_data : xr.DataArray
        All years data (for climatology statistics)
    time_dim : str
        Name of time dimension
    alpha : float
        Significance level (default 0.05 for 95% confidence)
        
    Returns
    -------
    tuple
        (t_statistic, p_value) DataArrays
    """
    # Get the climatological mean
    clim_mean = all_data.mean(dim=time_dim)
    
    # Stack spatial dimensions for vectorized computation
    composite_stacked = composite_data.stack(spatial=('latitude', 'longitude'))
    clim_mean_stacked = clim_mean.stack(spatial=('latitude', 'longitude'))
    
    # Number of samples in composite
    n_composite = composite_data[time_dim].size
    
    # Calculate t-statistic and p-value using one-sample t-test
    # H0: mean of composite years = climatological mean
    composite_mean = composite_stacked.mean(dim=time_dim)
    composite_std = composite_stacked.std(dim=time_dim, ddof=1)
    
    # t = (sample_mean - population_mean) / (sample_std / sqrt(n))
    t_stat = (composite_mean - clim_mean_stacked) / (composite_std / np.sqrt(n_composite))
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat.values), df=n_composite - 1))
    
    # Convert back to DataArray
    p_value_da = xr.DataArray(
        p_value,
        coords=composite_mean.coords,
        dims=composite_mean.dims
    )
    
    # Unstack
    t_stat = t_stat.unstack('spatial')
    p_value_da = p_value_da.unstack('spatial')
    
    return t_stat, p_value_da


def plot_anomaly_panels(anomalies: Dict[str, Dict[str, xr.DataArray]],
                        p_values: Dict[str, Dict[str, xr.DataArray]],
                        output_path: Path,
                        early_years: List[int],
                        late_years: List[int],
                        alpha: float = 0.05,
                        region: Tuple[float, float, float, float] = None,
                        cmap: str = 'RdBu_r',
                        figsize: Tuple[int, int] = (14, 10)) -> None:
    """
    Create multi-panel plot of VIMSE anomalies.
    
    Layout: 2 rows (April, May) x 2 columns (Early, Late)
    
    Parameters
    ----------
    anomalies : dict
        Nested dict: anomalies[month][onset_type] = DataArray
    p_values : dict
        Nested dict: p_values[month][onset_type] = DataArray
    output_path : Path
        Output PDF path
    alpha : float
        Significance level for stippling
    region : tuple
        (lon_min, lon_max, lat_min, lat_max) for plot extent
    cmap : str
        Colormap name
    figsize : tuple
        Figure size
    """
    # Create figure with subplots
    fig, axes = plt.subplots(
        2, 2, 
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    
    months = [4, 5]  # April, May
    onset_types = ['early', 'late']
    titles = {
        'early': 'Early Onset Years',
        'late': 'Late Onset Years'
    }
    
    # Determine common color scale
    all_values = []
    for month in months:
        for onset_type in onset_types:
            data = anomalies[month][onset_type]
            all_values.extend(data.values.flatten()[~np.isnan(data.values.flatten())])
    
    vmax = np.percentile(np.abs(all_values), 98)
    vmin = -vmax
    
    # Convert to MJ/m^2 for better readability
    scale_factor = 1e-6  # J/m^2 to MJ/m^2
    vmax_scaled = vmax * scale_factor
    vmin_scaled = vmin * scale_factor
    
    for i, month in enumerate(months):
        for j, onset_type in enumerate(onset_types):
            ax = axes[i, j]
            
            data = anomalies[month][onset_type] * scale_factor
            pval = p_values[month][onset_type]
            
            # Get coordinates
            if 'longitude' in data.coords:
                lon = data.longitude.values
                lat = data.latitude.values
            else:
                lon = data.lon.values
                lat = data.lat.values
            
            # Plot filled contours
            cf = ax.contourf(
                lon, lat, data.values,
                levels=np.linspace(vmin_scaled, vmax_scaled, 21),
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend='both'
            )
            
            # Add stippling for significant areas
            sig_mask = pval.values < alpha
            
            # Subsample for cleaner stippling
            skip = 4  # Plot every 4th point
            lon_mesh, lat_mesh = np.meshgrid(lon, lat)
            
            ax.scatter(
                lon_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                lat_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                marker='.', s=0.5, c='black', alpha=0.5,
                transform=ccrs.PlateCarree()
            )
            
            # Add coastlines and features
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
            ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.3)
            
            # Set extent if specified
            if region:
                ax.set_extent(region, crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            
            # Title
            ax.set_title(f"{MONTH_NAMES[month]} - {titles[onset_type]}", fontsize=12, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(
        cf, ax=axes, orientation='horizontal', 
        fraction=0.05, pad=0.08, aspect=40,
        label='VIMSE Anomaly (MJ/m²)'
    )
    
    # Add figure title
    fig.suptitle(
        'Vertically Integrated Moist Static Energy Anomalies\n'
        f'Early Onset Years: {early_years}\n'
        f'Late Onset Years: {late_years}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    # Save figure
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")


def save_results_netcdf(anomalies: Dict[str, Dict[str, xr.DataArray]],
                        p_values: Dict[str, Dict[str, xr.DataArray]],
                        climatologies: Dict[str, xr.DataArray],
                        output_path: Path,
                        early_years: List[int],
                        late_years: List[int]) -> None:
    """
    Save analysis results to NetCDF file.
    
    Parameters
    ----------
    anomalies : dict
        Nested dict of anomaly DataArrays
    p_values : dict
        Nested dict of p-value DataArrays
    climatologies : dict
        Dict of climatology DataArrays
    output_path : Path
        Output file path
    early_years : list
        List of early onset years
    late_years : list
        List of late onset years
    """
    ds = xr.Dataset()
    
    for month in [4, 5]:
        month_name = MONTH_NAMES[month].lower()
        
        # Climatology
        ds[f'vimse_clim_{month_name}'] = climatologies[month]
        ds[f'vimse_clim_{month_name}'].attrs = {
            'long_name': f'VIMSE Climatology for {MONTH_NAMES[month]}',
            'units': 'J m-2',
            'period': f'{CLIM_START_YEAR}-{CLIM_END_YEAR}'
        }
        
        for onset_type in ['early', 'late']:
            # Anomaly
            ds[f'vimse_anom_{month_name}_{onset_type}'] = anomalies[month][onset_type]
            ds[f'vimse_anom_{month_name}_{onset_type}'].attrs = {
                'long_name': f'VIMSE Anomaly for {MONTH_NAMES[month]} ({onset_type} onset)',
                'units': 'J m-2',
                'years': str(early_years if onset_type == 'early' else late_years)
            }
            
            # P-value
            ds[f'pvalue_{month_name}_{onset_type}'] = p_values[month][onset_type]
            ds[f'pvalue_{month_name}_{onset_type}'].attrs = {
                'long_name': f'T-test p-value for {MONTH_NAMES[month]} ({onset_type} onset)',
                'units': '1',
                'test': 'one-sample t-test against climatology'
            }
    
    ds.attrs = {
        'title': 'VIMSE Anomaly Analysis for Early/Late Monsoon Onset',
        'early_onset_years': str(early_years),
        'late_onset_years': str(late_years),
        'climatology_period': f'{CLIM_START_YEAR}-{CLIM_END_YEAR}',
        'Conventions': 'CF-1.7'
    }
    
    ds.to_netcdf(output_path)
    print(f"Results saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze VIMSE anomalies for early/late monsoon onset years.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using directory of yearly VIMSE files
  python analyze_vimse_anomaly.py /path/to/vimse_data/ --output-plot anomaly.pdf

  # Using single combined file
  python analyze_vimse_anomaly.py /path/to/vimse_combined.nc --single-file --output-plot anomaly.pdf

  # Specify region for plotting
  python analyze_vimse_anomaly.py /path/to/data/ --output-plot anomaly.pdf --region 40 160 -20 40

  # Custom onset years
  python analyze_vimse_anomaly.py /path/to/data/ --output-plot anomaly.pdf \\
      --early-years 1984,1985,1999,2000,2009,2017 \\
      --late-years 1983,1987,1993,1997,2010,2016,2018,2020,2021

Default onset years:
  Early: {early}
  Late: {late}
        """.format(early=EARLY_ONSET_YEARS, late=LATE_ONSET_YEARS)
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input path: directory with yearly VIMSE files or single combined file'
    )
    
    parser.add_argument(
        '--single-file',
        action='store_true',
        help='Input is a single combined NetCDF file'
    )
    
    parser.add_argument(
        '--output-plot',
        type=str,
        default='vimse_anomaly.pdf',
        help='Output PDF path (default: vimse_anomaly.pdf)'
    )
    
    parser.add_argument(
        '--output-nc',
        type=str,
        default=None,
        help='Output NetCDF path for results (optional)'
    )
    
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc',
        help='File pattern with {year} placeholder (for directory input)'
    )
    
    parser.add_argument(
        '--vimse-var',
        type=str,
        default='vimse',
        help='Name of VIMSE variable in files (default: vimse)'
    )
    
    parser.add_argument(
        '--region',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
        default=None,
        help='Plot region extent: lon_min lon_max lat_min lat_max'
    )
    
    parser.add_argument(
        '--early-years',
        type=str,
        default=None,
        help='Comma-separated list of early onset years'
    )
    
    parser.add_argument(
        '--late-years',
        type=str,
        default=None,
        help='Comma-separated list of late onset years'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level for t-test (default: 0.05)'
    )
    
    args = parser.parse_args()
    
    # Parse onset years
    if args.early_years:
        early_years = [int(y.strip()) for y in args.early_years.split(',')]
    else:
        early_years = list(EARLY_ONSET_YEARS)
    
    if args.late_years:
        late_years = [int(y.strip()) for y in args.late_years.split(',')]
    else:
        late_years = list(LATE_ONSET_YEARS)
    
    print("=" * 60)
    print("VIMSE Anomaly Analysis for Monsoon Onset")
    print("=" * 60)
    print(f"Early onset years: {early_years}")
    print(f"Late onset years: {late_years}")
    print(f"Climatology period: {CLIM_START_YEAR}-{CLIM_END_YEAR}")
    print(f"Significance level: {args.alpha}")
    print()
    
    # Load data
    input_path = Path(args.input)
    print(f"Loading VIMSE data from: {input_path}")
    
    if args.single_file:
        vimse = load_vimse_from_single_file(input_path, args.vimse_var)
    else:
        vimse = load_vimse_data(input_path, args.file_pattern, 
                               CLIM_START_YEAR, CLIM_END_YEAR, args.vimse_var)
    
    # Detect time dimension
    time_dim = 'valid_time' if 'valid_time' in vimse.dims else 'time'
    print(f"Time dimension: {time_dim}")
    print(f"Data shape: {vimse.shape}")
    print()
    
    # Process each month
    anomalies = {}
    p_values = {}
    climatologies = {}
    
    for month in ANALYSIS_MONTHS:
        print(f"Processing {MONTH_NAMES[month]}...")
        
        # Extract month data
        vimse_month = extract_month_data(vimse, month, time_dim)
        print(f"  Found {vimse_month[time_dim].size} time steps for {MONTH_NAMES[month]}")
        
        # Calculate climatology
        climatology = calculate_climatology(vimse_month, CLIM_START_YEAR, CLIM_END_YEAR, time_dim)
        climatologies[month] = climatology
        
        anomalies[month] = {}
        p_values[month] = {}
        
        # Calculate anomalies for early and late onset years
        for onset_type, years in [('early', early_years), ('late', late_years)]:
            print(f"  Computing {onset_type} onset composite ({len(years)} years)...")
            
            # Calculate composite anomaly
            anomaly, composite_data = calculate_composite_anomaly(
                vimse_month, climatology, years, time_dim
            )
            anomalies[month][onset_type] = anomaly
            
            # Perform t-test
            t_stat, pval = perform_ttest(composite_data, vimse_month, time_dim, args.alpha)
            p_values[month][onset_type] = pval
            
            # Print statistics
            n_sig = (pval < args.alpha).sum().values
            total = pval.size
            print(f"    Significant grid points: {n_sig}/{total} ({100*n_sig/total:.1f}%)")
    
    print()
    
    # Plot results
    print("Creating visualization...")
    region = tuple(args.region) if args.region else None
    
    plot_anomaly_panels(
        anomalies, p_values, 
        Path(args.output_plot),
        early_years=early_years,
        late_years=late_years,
        alpha=args.alpha,
        region=region
    )
    
    # Save NetCDF results if requested
    if args.output_nc:
        print("Saving results to NetCDF...")
        save_results_netcdf(anomalies, p_values, climatologies, Path(args.output_nc),
                           early_years=early_years, late_years=late_years)
    
    print()
    print("=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()