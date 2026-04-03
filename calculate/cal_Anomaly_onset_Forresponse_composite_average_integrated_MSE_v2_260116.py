"""
260116
Analyze Vertically Integrated MSE (VIMSE) and component anomalies for 
early and late monsoon onset years.

This script:
1. Reads VIMSE and component data (vicpt, viphi, vilvq) from NetCDF files
2. Calculates climatology for March and April (1980-2021)
3. Computes composite anomalies for early and late onset years
4. Performs Student's t-test for statistical significance
5. Visualizes results with stippling for significant areas

Components:
- VIMSE: Vertically Integrated Moist Static Energy
- VICPT: Vertically Integrated cp*T (sensible heat term)
- VIPHI: Vertically Integrated Phi (geopotential term)
- VILVQ: Vertically Integrated Lv*q (latent heat term)

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

# Months to analyze (March=3, April=4)
ANALYSIS_MONTHS = [3, 4]
MONTH_NAMES = {3: 'March', 4: 'April'}

# Variables to analyze
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


def load_all_variables(data_path: Path, 
                       file_pattern: str = "ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc",
                       start_year: int = CLIM_START_YEAR,
                       end_year: int = CLIM_END_YEAR,
                       variables: List[str] = VARIABLES) -> Dict[str, xr.DataArray]:
    """
    Load VIMSE and component data from multiple yearly files.
    
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
    variables : list
        List of variable names to load
        
    Returns
    -------
    dict
        Dictionary mapping variable names to combined DataArrays
    """
    data_dict = {var: [] for var in variables}
    
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
            for var in variables:
                if var in ds:
                    data_dict[var].append(ds[var])
                else:
                    warnings.warn(f"Variable '{var}' not found in {filepath}")
            ds.close()
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            continue
    
    # Concatenate along time dimension
    result = {}
    for var in variables:
        if data_dict[var]:
            result[var] = xr.concat(data_dict[var], dim='valid_time')
        else:
            warnings.warn(f"No data found for variable '{var}'")
    
    return result


def load_from_single_file(filepath: Path, 
                          variables: List[str] = VARIABLES) -> Dict[str, xr.DataArray]:
    """
    Load VIMSE and component data from a single combined NetCDF file.
    
    Parameters
    ----------
    filepath : Path
        Path to combined VIMSE NetCDF file
    variables : list
        List of variable names to load
        
    Returns
    -------
    dict
        Dictionary mapping variable names to DataArrays
    """
    ds = xr.open_dataset(filepath)
    
    result = {}
    for var in variables:
        if var in ds:
            result[var] = ds[var]
        else:
            warnings.warn(f"Variable '{var}' not found in {filepath}")
    
    return result


def extract_month_data(data: xr.DataArray, 
                       month: int,
                       time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Extract data for a specific month across all years.
    
    Parameters
    ----------
    data : xr.DataArray
        Data with time dimension
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
    time_coord = data[time_dim]
    
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
    
    data_month = data.sel({time_dim: month_mask})
    
    # Add year coordinate for easier selection
    years = data_month[time_dim].dt.year.values
    data_month = data_month.assign_coords(year=(time_dim, years))
    
    return data_month


def calculate_climatology(data_month: xr.DataArray,
                          start_year: int = CLIM_START_YEAR,
                          end_year: int = CLIM_END_YEAR,
                          time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Calculate climatological mean for a specific month.
    
    Parameters
    ----------
    data_month : xr.DataArray
        Data for a specific month
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
    years = data_month[time_dim].dt.year
    year_mask = (years >= start_year) & (years <= end_year)
    data_clim_period = data_month.sel({time_dim: year_mask})
    
    # Calculate mean
    climatology = data_clim_period.mean(dim=time_dim)
    
    return climatology


def calculate_composite_anomaly(data_month: xr.DataArray,
                                climatology: xr.DataArray,
                                years: List[int],
                                time_dim: str = 'valid_time') -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate composite mean anomaly for selected years.
    
    Parameters
    ----------
    data_month : xr.DataArray
        Data for a specific month
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
    year_values = data_month[time_dim].dt.year.values
    year_mask = np.isin(year_values, years)
    
    if not np.any(year_mask):
        raise ValueError(f"No data found for years: {years}")
    
    data_composite = data_month.isel({time_dim: year_mask})
    
    # Calculate composite mean
    composite_mean = data_composite.mean(dim=time_dim)
    
    # Calculate anomaly
    anomaly = composite_mean - climatology
    
    return anomaly, data_composite


def perform_ttest(composite_data: xr.DataArray,
                  all_data: xr.DataArray,
                  time_dim: str = 'valid_time',
                  alpha: float = 0.10) -> Tuple[xr.DataArray, xr.DataArray]:
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
        Significance level (default 0.10 for 90% confidence)
        
    Returns
    -------
    tuple
        (t_statistic, p_value) DataArrays
    """
    # Get the climatological mean
    clim_mean = all_data.mean(dim=time_dim)
    
    # Detect lat/lon dimension names
    lat_dim = 'latitude' if 'latitude' in composite_data.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in composite_data.dims else 'lon'
    
    # Stack spatial dimensions for vectorized computation
    composite_stacked = composite_data.stack(spatial=(lat_dim, lon_dim))
    clim_mean_stacked = clim_mean.stack(spatial=(lat_dim, lon_dim))
    
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


def plot_single_variable(anomalies: Dict[int, Dict[str, xr.DataArray]],
                         p_values: Dict[int, Dict[str, xr.DataArray]],
                         output_path: Path,
                         var_name: str,
                         early_years: List[int],
                         late_years: List[int],
                         alpha: float = 0.10,
                         region: Tuple[float, float, float, float] = (45, 115, -10, 30),
                         cmap: str = 'RdBu_r',
                         interval: float = 1.5,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create 2x2 panel plot for a single variable (2 months x 2 onset types).
    
    Parameters
    ----------
    anomalies : dict
        Nested dict: anomalies[month][onset_type] = DataArray
    p_values : dict
        Nested dict: p_values[month][onset_type] = DataArray
    output_path : Path
        Output PDF path
    var_name : str
        Variable name for title
    early_years : list
        List of early onset years
    late_years : list
        List of late onset years
    alpha : float
        Significance level for stippling
    region : tuple
        (lon_min, lon_max, lat_min, lat_max) for plot extent
    cmap : str
        Colormap name
    interval : float
        Colorbar interval in MJ/m²
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
    
    months = ANALYSIS_MONTHS
    onset_types = ['early', 'late']
    titles = {
        'early': 'Early Onset',
        'late': 'Late Onset'
    }
    
    # Convert to MJ/m^2 for better readability
    scale_factor = 1e-6  # J/m^2 to MJ/m^2
    
    # Determine color scale from data
    all_values = []
    for month in months:
        for onset_type in onset_types:
            data = anomalies[month][onset_type]
            all_values.extend(data.values.flatten()[~np.isnan(data.values.flatten())])
    
    vmax_data = np.percentile(np.abs(all_values), 98) * scale_factor
    vmax_scaled = np.ceil(vmax_data / interval) * interval
    if vmax_scaled == 0:
        vmax_scaled = interval
    vmin_scaled = -vmax_scaled
    levels = np.arange(vmin_scaled, vmax_scaled + interval, interval)
    
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
            
            # Subset to region for plotting
            lon_mask = (lon >= region[0]) & (lon <= region[1])
            lat_mask = (lat >= region[2]) & (lat <= region[3])
            
            lon_sub = lon[lon_mask]
            lat_sub = lat[lat_mask]
            data_sub = data.values[np.ix_(lat_mask, lon_mask)]
            pval_sub = pval.values[np.ix_(lat_mask, lon_mask)]
            
            # Plot filled contours
            cf = ax.contourf(
                lon_sub, lat_sub, data_sub,
                levels=levels,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend='both'
            )
            
            # Add stippling for significant areas
            sig_mask = pval_sub < alpha
            
            # Subsample for cleaner stippling
            skip = 4
            lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
            
            ax.scatter(
                lon_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                lat_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                marker='.', s=0.5, c='black', alpha=0.5,
                transform=ccrs.PlateCarree()
            )
            
            # Add coastlines and features
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
            
            # Set extent
            ax.set_extent(region, crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            
            # Title
            ax.set_title(f"{MONTH_NAMES[month]} - {titles[onset_type]}", fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(
        cf, ax=axes, orientation='horizontal', 
        fraction=0.05, pad=0.08, aspect=40,
        label=f'{VARIABLE_NAMES.get(var_name, var_name)} Anomaly (MJ/m²)'
    )
    
    # Add figure title
    confidence_pct = int((1 - alpha) * 100)
    fig.suptitle(
        f'{VARIABLE_LONG_NAMES.get(var_name, var_name)} Anomalies\n'
        f'Early: {early_years} | Late: {late_years}\n'
        f'(Stippling: {confidence_pct}% significance)',
        fontsize=12, fontweight='bold', y=1.02
    )
    
    # Save figure
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure saved to: {output_path}")


def plot_all_variables_combined(all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                                all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                                output_path: Path,
                                early_years: List[int],
                                late_years: List[int],
                                alpha: float = 0.10,
                                region: Tuple[float, float, float, float] = (45, 115, -10, 30),
                                cmap: str = 'RdBu_r',
                                interval: float = 1.5) -> None:
    """
    Create a comprehensive multi-panel plot showing all variables.
    
    Layout: 4 rows (VIMSE, VICPT, VIPHI, VILVQ) x 4 columns (Mar-Early, Mar-Late, Apr-Early, Apr-Late)
    
    Parameters
    ----------
    all_anomalies : dict
        Nested dict: all_anomalies[var][month][onset_type] = DataArray
    all_pvalues : dict
        Nested dict: all_pvalues[var][month][onset_type] = DataArray
    output_path : Path
        Output PDF path
    early_years : list
        List of early onset years
    late_years : list
        List of late onset years
    alpha : float
        Significance level for stippling
    region : tuple
        Plot region extent
    cmap : str
        Colormap name
    interval : float
        Colorbar interval in MJ/m²
    """
    variables = VARIABLES
    months = ANALYSIS_MONTHS
    onset_types = ['early', 'late']
    
    n_rows = len(variables)
    n_cols = len(months) * len(onset_types)  # 4 columns
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 14),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    
    scale_factor = 1e-6  # J/m^2 to MJ/m^2
    
    # Store contour fills for colorbars (one per row)
    cf_per_row = {}
    levels_per_row = {}
    
    for row, var in enumerate(variables):
        # Determine color scale for this variable
        all_values = []
        for month in months:
            for onset_type in onset_types:
                if var in all_anomalies and month in all_anomalies[var]:
                    data = all_anomalies[var][month][onset_type]
                    all_values.extend(data.values.flatten()[~np.isnan(data.values.flatten())])
        
        if not all_values:
            continue
            
        vmax_data = np.percentile(np.abs(all_values), 98) * scale_factor
        vmax_scaled = np.ceil(vmax_data / interval) * interval
        if vmax_scaled == 0:
            vmax_scaled = interval
        vmin_scaled = -vmax_scaled
        levels = np.arange(vmin_scaled, vmax_scaled + interval, interval)
        levels_per_row[row] = levels
        
        col = 0
        for month in months:
            for onset_type in onset_types:
                ax = axes[row, col]
                
                if var not in all_anomalies or month not in all_anomalies[var]:
                    ax.set_visible(False)
                    col += 1
                    continue
                
                data = all_anomalies[var][month][onset_type] * scale_factor
                pval = all_pvalues[var][month][onset_type]
                
                # Get coordinates
                if 'longitude' in data.coords:
                    lon = data.longitude.values
                    lat = data.latitude.values
                else:
                    lon = data.lon.values
                    lat = data.lat.values
                
                # Subset to region
                lon_mask = (lon >= region[0]) & (lon <= region[1])
                lat_mask = (lat >= region[2]) & (lat <= region[3])
                
                lon_sub = lon[lon_mask]
                lat_sub = lat[lat_mask]
                data_sub = data.values[np.ix_(lat_mask, lon_mask)]
                pval_sub = pval.values[np.ix_(lat_mask, lon_mask)]
                
                # Plot
                cf = ax.contourf(
                    lon_sub, lat_sub, data_sub,
                    levels=levels,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )
                
                # Store for colorbar (use last column)
                if col == n_cols - 1:
                    cf_per_row[row] = cf
                
                # Stippling
                sig_mask = pval_sub < alpha
                skip = 5
                lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
                ax.scatter(
                    lon_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                    lat_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                    marker='.', s=0.3, c='black', alpha=0.5,
                    transform=ccrs.PlateCarree()
                )
                
                ax.coastlines(linewidth=0.4)
                ax.set_extent(region, crs=ccrs.PlateCarree())
                
                # Gridlines
                gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle='--')
                if col == 0:
                    gl.left_labels = True
                    gl.yformatter = LatitudeFormatter()
                if row == n_rows - 1:
                    gl.bottom_labels = True
                    gl.xformatter = LongitudeFormatter()
                
                # Column titles (top row only)
                if row == 0:
                    onset_label = 'Early' if onset_type == 'early' else 'Late'
                    ax.set_title(f"{MONTH_NAMES[month]}\n{onset_label}", fontsize=10, fontweight='bold')
                
                # Row labels (left side)
                if col == 0:
                    ax.text(-0.15, 0.5, VARIABLE_NAMES[var], transform=ax.transAxes,
                           fontsize=10, fontweight='bold', va='center', ha='right',
                           rotation=90)
                
                col += 1
    
    # Add colorbars on the right side for each row
    for row in range(n_rows):
        if row in cf_per_row:
            # Position colorbar to the right of each row
            cax = fig.add_axes([0.92, 0.78 - row*0.21, 0.012, 0.15])
            cbar = fig.colorbar(cf_per_row[row], cax=cax, orientation='vertical')
            cbar.set_label('MJ/m²', fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    
    # Main title
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


def save_results_netcdf(all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                        all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                        all_climatologies: Dict[str, Dict[int, xr.DataArray]],
                        output_path: Path,
                        early_years: List[int],
                        late_years: List[int]) -> None:
    """
    Save analysis results to NetCDF file.
    """
    ds = xr.Dataset()
    
    for var in VARIABLES:
        if var not in all_anomalies:
            continue
            
        for month in ANALYSIS_MONTHS:
            if month not in all_anomalies[var]:
                continue
                
            month_name = MONTH_NAMES[month].lower()
            
            # Climatology
            if var in all_climatologies and month in all_climatologies[var]:
                ds[f'{var}_clim_{month_name}'] = all_climatologies[var][month]
                ds[f'{var}_clim_{month_name}'].attrs = {
                    'long_name': f'{VARIABLE_LONG_NAMES[var]} Climatology for {MONTH_NAMES[month]}',
                    'units': 'J m-2',
                    'period': f'{CLIM_START_YEAR}-{CLIM_END_YEAR}'
                }
            
            for onset_type in ['early', 'late']:
                years_list = early_years if onset_type == 'early' else late_years
                
                # Anomaly
                ds[f'{var}_anom_{month_name}_{onset_type}'] = all_anomalies[var][month][onset_type]
                ds[f'{var}_anom_{month_name}_{onset_type}'].attrs = {
                    'long_name': f'{VARIABLE_LONG_NAMES[var]} Anomaly for {MONTH_NAMES[month]} ({onset_type} onset)',
                    'units': 'J m-2',
                    'years': str(years_list)
                }
                
                # P-value
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze VIMSE and component anomalies for early/late monsoon onset years.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using directory of yearly VIMSE files
  python analyze_vimse_anomaly.py /path/to/vimse_data/ --output-dir ./output/

  # Using single combined file
  python analyze_vimse_anomaly.py /path/to/vimse_combined.nc --single-file --output-dir ./output/

  # Specify region for plotting
  python analyze_vimse_anomaly.py /path/to/data/ --output-dir ./output/ --region 60 120 -10 40

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
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for PDF plots (default: current directory)'
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
        '--region',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
        default=[45, 115, -10, 30],
        help='Plot region extent (default: 45 115 -10 30)'
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
        default=0.10,
        help='Significance level for t-test (default: 0.10 for 90%% confidence)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=1.5,
        help='Colorbar interval in MJ/m² (default: 1.5)'
    )
    
    parser.add_argument(
        '--no-individual',
        action='store_true',
        help='Skip individual variable plots, only create combined plot'
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
    
    print("=" * 70)
    print("VIMSE and Component Anomaly Analysis for Monsoon Onset")
    print("=" * 70)
    print(f"Early onset years: {early_years}")
    print(f"Late onset years: {late_years}")
    print(f"Climatology period: {CLIM_START_YEAR}-{CLIM_END_YEAR}")
    print(f"Analysis months: {[MONTH_NAMES[m] for m in ANALYSIS_MONTHS]}")
    print(f"Significance level: {args.alpha} ({int((1-args.alpha)*100)}% confidence)")
    print(f"Variables: {VARIABLES}")
    print()
    
    # Load data
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
    
    # Detect time dimension
    sample_var = list(data_dict.values())[0]
    time_dim = 'valid_time' if 'valid_time' in sample_var.dims else 'time'
    print(f"Time dimension: {time_dim}")
    print()
    
    # Process each variable
    all_anomalies = {}
    all_pvalues = {}
    all_climatologies = {}
    
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
            
            # Extract month data
            data_month = extract_month_data(data, month, time_dim)
            
            # Calculate climatology
            climatology = calculate_climatology(data_month, CLIM_START_YEAR, CLIM_END_YEAR, time_dim)
            all_climatologies[var][month] = climatology
            
            all_anomalies[var][month] = {}
            all_pvalues[var][month] = {}
            
            # Calculate anomalies for early and late onset years
            for onset_type, years in [('early', early_years), ('late', late_years)]:
                # Calculate composite anomaly
                anomaly, composite_data = calculate_composite_anomaly(
                    data_month, climatology, years, time_dim
                )
                all_anomalies[var][month][onset_type] = anomaly
                
                # Perform t-test
                t_stat, pval = perform_ttest(composite_data, data_month, time_dim, args.alpha)
                all_pvalues[var][month][onset_type] = pval
                
                # Print statistics
                n_sig = (pval < args.alpha).sum().values
                total = pval.size
                print(f"    {onset_type.capitalize()}: {n_sig}/{total} significant ({100*n_sig/total:.1f}%)")
    
    print()
    
    # Create plots
    region = tuple(args.region)
    
    # Individual variable plots
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
    
    # Combined plot
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
    
    # Save NetCDF results if requested
    if args.output_nc:
        print("Saving results to NetCDF...")
        save_results_netcdf(all_anomalies, all_pvalues, all_climatologies, 
                           Path(args.output_nc), early_years, late_years)
    
    print()
    print("=" * 70)
    print("Analysis complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Analyze Vertically Integrated MSE (VIMSE) and component anomalies for 
early and late monsoon onset years.

This script:
1. Reads VIMSE and component data (vicpt, viphi, vilvq) from NetCDF files
2. Calculates climatology for March and April (1980-2021)
3. Computes composite anomalies for early and late onset years
4. Performs Student's t-test for statistical significance
5. Visualizes results with stippling for significant areas

Components:
- VIMSE: Vertically Integrated Moist Static Energy
- VICPT: Vertically Integrated cp*T (sensible heat term)
- VIPHI: Vertically Integrated Phi (geopotential term)
- VILVQ: Vertically Integrated Lv*q (latent heat term)

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

# Months to analyze (March=3, April=4)
ANALYSIS_MONTHS = [3, 4]
MONTH_NAMES = {3: 'March', 4: 'April'}

# Variables to analyze
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


def load_all_variables(data_path: Path, 
                       file_pattern: str = "ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc",
                       start_year: int = CLIM_START_YEAR,
                       end_year: int = CLIM_END_YEAR,
                       variables: List[str] = VARIABLES) -> Dict[str, xr.DataArray]:
    """
    Load VIMSE and component data from multiple yearly files.
    
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
    variables : list
        List of variable names to load
        
    Returns
    -------
    dict
        Dictionary mapping variable names to combined DataArrays
    """
    data_dict = {var: [] for var in variables}
    
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
            for var in variables:
                if var in ds:
                    data_dict[var].append(ds[var])
                else:
                    warnings.warn(f"Variable '{var}' not found in {filepath}")
            ds.close()
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            continue
    
    # Concatenate along time dimension
    result = {}
    for var in variables:
        if data_dict[var]:
            result[var] = xr.concat(data_dict[var], dim='valid_time')
        else:
            warnings.warn(f"No data found for variable '{var}'")
    
    return result


def load_from_single_file(filepath: Path, 
                          variables: List[str] = VARIABLES) -> Dict[str, xr.DataArray]:
    """
    Load VIMSE and component data from a single combined NetCDF file.
    
    Parameters
    ----------
    filepath : Path
        Path to combined VIMSE NetCDF file
    variables : list
        List of variable names to load
        
    Returns
    -------
    dict
        Dictionary mapping variable names to DataArrays
    """
    ds = xr.open_dataset(filepath)
    
    result = {}
    for var in variables:
        if var in ds:
            result[var] = ds[var]
        else:
            warnings.warn(f"Variable '{var}' not found in {filepath}")
    
    return result


def extract_month_data(data: xr.DataArray, 
                       month: int,
                       time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Extract data for a specific month across all years.
    
    Parameters
    ----------
    data : xr.DataArray
        Data with time dimension
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
    time_coord = data[time_dim]
    
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
    
    data_month = data.sel({time_dim: month_mask})
    
    # Add year coordinate for easier selection
    years = data_month[time_dim].dt.year.values
    data_month = data_month.assign_coords(year=(time_dim, years))
    
    return data_month


def calculate_climatology(data_month: xr.DataArray,
                          start_year: int = CLIM_START_YEAR,
                          end_year: int = CLIM_END_YEAR,
                          time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Calculate climatological mean for a specific month.
    
    Parameters
    ----------
    data_month : xr.DataArray
        Data for a specific month
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
    years = data_month[time_dim].dt.year
    year_mask = (years >= start_year) & (years <= end_year)
    data_clim_period = data_month.sel({time_dim: year_mask})
    
    # Calculate mean
    climatology = data_clim_period.mean(dim=time_dim)
    
    return climatology


def calculate_composite_anomaly(data_month: xr.DataArray,
                                climatology: xr.DataArray,
                                years: List[int],
                                time_dim: str = 'valid_time') -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Calculate composite mean anomaly for selected years.
    
    Parameters
    ----------
    data_month : xr.DataArray
        Data for a specific month
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
    year_values = data_month[time_dim].dt.year.values
    year_mask = np.isin(year_values, years)
    
    if not np.any(year_mask):
        raise ValueError(f"No data found for years: {years}")
    
    data_composite = data_month.isel({time_dim: year_mask})
    
    # Calculate composite mean
    composite_mean = data_composite.mean(dim=time_dim)
    
    # Calculate anomaly
    anomaly = composite_mean - climatology
    
    return anomaly, data_composite


def perform_ttest(composite_data: xr.DataArray,
                  all_data: xr.DataArray,
                  time_dim: str = 'valid_time',
                  alpha: float = 0.10) -> Tuple[xr.DataArray, xr.DataArray]:
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
        Significance level (default 0.10 for 90% confidence)
        
    Returns
    -------
    tuple
        (t_statistic, p_value) DataArrays
    """
    # Get the climatological mean
    clim_mean = all_data.mean(dim=time_dim)
    
    # Detect lat/lon dimension names
    lat_dim = 'latitude' if 'latitude' in composite_data.dims else 'lat'
    lon_dim = 'longitude' if 'longitude' in composite_data.dims else 'lon'
    
    # Stack spatial dimensions for vectorized computation
    composite_stacked = composite_data.stack(spatial=(lat_dim, lon_dim))
    clim_mean_stacked = clim_mean.stack(spatial=(lat_dim, lon_dim))
    
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


def plot_single_variable(anomalies: Dict[int, Dict[str, xr.DataArray]],
                         p_values: Dict[int, Dict[str, xr.DataArray]],
                         output_path: Path,
                         var_name: str,
                         early_years: List[int],
                         late_years: List[int],
                         alpha: float = 0.10,
                         region: Tuple[float, float, float, float] = (45, 115, -10, 30),
                         cmap: str = 'RdBu_r',
                         interval: float = 1.5,
                         figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create 2x2 panel plot for a single variable (2 months x 2 onset types).
    
    Parameters
    ----------
    anomalies : dict
        Nested dict: anomalies[month][onset_type] = DataArray
    p_values : dict
        Nested dict: p_values[month][onset_type] = DataArray
    output_path : Path
        Output PDF path
    var_name : str
        Variable name for title
    early_years : list
        List of early onset years
    late_years : list
        List of late onset years
    alpha : float
        Significance level for stippling
    region : tuple
        (lon_min, lon_max, lat_min, lat_max) for plot extent
    cmap : str
        Colormap name
    interval : float
        Colorbar interval in MJ/m²
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
    
    months = ANALYSIS_MONTHS
    onset_types = ['early', 'late']
    titles = {
        'early': 'Early Onset',
        'late': 'Late Onset'
    }
    
    # Convert to MJ/m^2 for better readability
    scale_factor = 1e-6  # J/m^2 to MJ/m^2
    
    # Determine color scale from data
    all_values = []
    for month in months:
        for onset_type in onset_types:
            data = anomalies[month][onset_type]
            all_values.extend(data.values.flatten()[~np.isnan(data.values.flatten())])
    
    vmax_data = np.percentile(np.abs(all_values), 98) * scale_factor
    vmax_scaled = np.ceil(vmax_data / interval) * interval
    if vmax_scaled == 0:
        vmax_scaled = interval
    vmin_scaled = -vmax_scaled
    levels = np.arange(vmin_scaled, vmax_scaled + interval, interval)
    
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
            
            # Subset to region for plotting
            lon_mask = (lon >= region[0]) & (lon <= region[1])
            lat_mask = (lat >= region[2]) & (lat <= region[3])
            
            lon_sub = lon[lon_mask]
            lat_sub = lat[lat_mask]
            data_sub = data.values[np.ix_(lat_mask, lon_mask)]
            pval_sub = pval.values[np.ix_(lat_mask, lon_mask)]
            
            # Plot filled contours
            cf = ax.contourf(
                lon_sub, lat_sub, data_sub,
                levels=levels,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend='both'
            )
            
            # Add stippling for significant areas
            sig_mask = pval_sub < alpha
            
            # Subsample for cleaner stippling
            skip = 4
            lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
            
            ax.scatter(
                lon_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                lat_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                marker='.', s=0.5, c='black', alpha=0.5,
                transform=ccrs.PlateCarree()
            )
            
            # Add coastlines and features
            ax.coastlines(linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
            
            # Set extent
            ax.set_extent(region, crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, linewidth=0.5, alpha=0.5, linestyle='--')
            gl.top_labels = False
            gl.right_labels = False
            gl.xformatter = LongitudeFormatter()
            gl.yformatter = LatitudeFormatter()
            
            # Title
            ax.set_title(f"{MONTH_NAMES[month]} - {titles[onset_type]}", fontsize=11, fontweight='bold')
    
    # Add colorbar
    cbar = fig.colorbar(
        cf, ax=axes, orientation='horizontal', 
        fraction=0.05, pad=0.08, aspect=40,
        label=f'{VARIABLE_NAMES.get(var_name, var_name)} Anomaly (MJ/m²)'
    )
    
    # Add figure title
    confidence_pct = int((1 - alpha) * 100)
    fig.suptitle(
        f'{VARIABLE_LONG_NAMES.get(var_name, var_name)} Anomalies\n'
        f'Early: {early_years} | Late: {late_years}\n'
        f'(Stippling: {confidence_pct}% significance)',
        fontsize=12, fontweight='bold', y=1.02
    )
    
    # Save figure
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Figure saved to: {output_path}")


def plot_early_late_separate(all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                             all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                             output_path: Path,
                             onset_type: str,
                             onset_years: List[int],
                             alpha: float = 0.10,
                             region: Tuple[float, float, float, float] = (45, 115, -10, 30),
                             cmap: str = 'RdBu_r',
                             interval: float = 1.5) -> None:
    """
    Create a 4x3 panel plot for either early or late onset years.
    
    Layout: 4 rows (VIMSE, VICPT, VIPHI, VILVQ) x 3 columns (March, April, April-March)
    
    Parameters
    ----------
    all_anomalies : dict
        Nested dict: all_anomalies[var][month][onset_type] = DataArray
    all_pvalues : dict
        Nested dict: all_pvalues[var][month][onset_type] = DataArray
    output_path : Path
        Output PDF path
    onset_type : str
        'early' or 'late'
    onset_years : list
        List of onset years
    alpha : float
        Significance level for stippling
    region : tuple
        Plot region extent
    cmap : str
        Colormap name
    interval : float
        Colorbar interval in MJ/m²
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
    
    scale_factor = 1e-6  # J/m^2 to MJ/m^2
    
    # Store contour fills for colorbars (one per row)
    cf_per_row = {}
    
    for row, var in enumerate(variables):
        if var not in all_anomalies:
            continue
        
        # Get March and April data
        mar_data = all_anomalies[var][3][onset_type]
        apr_data = all_anomalies[var][4][onset_type]
        diff_data = apr_data - mar_data
        
        mar_pval = all_pvalues[var][3][onset_type]
        apr_pval = all_pvalues[var][4][onset_type]
        # For difference, use minimum p-value (conservative approach)
        diff_pval = np.minimum(mar_pval.values, apr_pval.values)
        
        data_list = [mar_data, apr_data, diff_data]
        pval_list = [mar_pval.values, apr_pval.values, diff_pval]
        
        # Determine color scale for this variable (across all 3 columns)
        all_values = []
        for d in data_list:
            all_values.extend(d.values.flatten()[~np.isnan(d.values.flatten())])
        
        vmax_data = np.percentile(np.abs(all_values), 98) * scale_factor
        vmax_scaled = np.ceil(vmax_data / interval) * interval
        if vmax_scaled == 0:
            vmax_scaled = interval
        vmin_scaled = -vmax_scaled
        levels = np.arange(vmin_scaled, vmax_scaled + interval, interval)
        
        for col in range(n_cols):
            ax = axes[row, col]
            
            data = data_list[col] * scale_factor
            pval = pval_list[col]
            
            # Get coordinates
            if 'longitude' in data.coords:
                lon = data.longitude.values
                lat = data.latitude.values
            else:
                lon = data.lon.values
                lat = data.lat.values
            
            # Subset to region
            lon_mask = (lon >= region[0]) & (lon <= region[1])
            lat_mask = (lat >= region[2]) & (lat <= region[3])
            
            lon_sub = lon[lon_mask]
            lat_sub = lat[lat_mask]
            data_sub = data.values[np.ix_(lat_mask, lon_mask)]
            pval_sub = pval[np.ix_(lat_mask, lon_mask)]
            
            # Plot
            cf = ax.contourf(
                lon_sub, lat_sub, data_sub,
                levels=levels,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend='both'
            )
            
            # Store for colorbar (use last column)
            if col == n_cols - 1:
                cf_per_row[row] = cf
            
            # Stippling
            sig_mask = pval_sub < alpha
            skip = 5
            lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
            ax.scatter(
                lon_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                lat_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                marker='.', s=0.4, c='black', alpha=0.5,
                transform=ccrs.PlateCarree()
            )
            
            ax.coastlines(linewidth=0.4)
            ax.set_extent(region, crs=ccrs.PlateCarree())
            
            # Gridlines
            gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle='--')
            if col == 0:
                gl.left_labels = True
                gl.yformatter = LatitudeFormatter()
            if row == n_rows - 1:
                gl.bottom_labels = True
                gl.xformatter = LongitudeFormatter()
            
            # Column titles (top row only)
            if row == 0:
                ax.set_title(columns[col], fontsize=11, fontweight='bold')
            
            # Row labels (left side)
            if col == 0:
                ax.text(-0.15, 0.5, VARIABLE_NAMES[var], transform=ax.transAxes,
                       fontsize=10, fontweight='bold', va='center', ha='right',
                       rotation=90)
    
    # Add colorbars on the right side for each row
    for row in range(n_rows):
        if row in cf_per_row:
            cax = fig.add_axes([0.92, 0.78 - row*0.21, 0.012, 0.15])
            cbar = fig.colorbar(cf_per_row[row], cax=cax, orientation='vertical')
            cbar.set_label('MJ/m²', fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    
    # Main title
    onset_label = 'Early' if onset_type == 'early' else 'Late'
    confidence_pct = int((1 - alpha) * 100)
    fig.suptitle(
        f'{onset_label} Onset Years: {onset_years}\n'
        f'Vertically Integrated MSE and Component Anomalies\n'
        f'(Stippling: {confidence_pct}% significance)',
        fontsize=13, fontweight='bold', y=1.02
    )
    
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  {onset_label} onset figure saved to: {output_path}")


def plot_all_variables_combined(all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                                all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                                output_path: Path,
                                early_years: List[int],
                                late_years: List[int],
                                alpha: float = 0.10,
                                region: Tuple[float, float, float, float] = (45, 115, -10, 30),
                                cmap: str = 'RdBu_r',
                                interval: float = 1.5) -> None:
    """
    Create a comprehensive multi-panel plot showing all variables.
    
    Layout: 4 rows (VIMSE, VICPT, VIPHI, VILVQ) x 4 columns (Mar-Early, Mar-Late, Apr-Early, Apr-Late)
    
    Parameters
    ----------
    all_anomalies : dict
        Nested dict: all_anomalies[var][month][onset_type] = DataArray
    all_pvalues : dict
        Nested dict: all_pvalues[var][month][onset_type] = DataArray
    output_path : Path
        Output PDF path
    early_years : list
        List of early onset years
    late_years : list
        List of late onset years
    alpha : float
        Significance level for stippling
    region : tuple
        Plot region extent
    cmap : str
        Colormap name
    interval : float
        Colorbar interval in MJ/m²
    """
    variables = VARIABLES
    months = ANALYSIS_MONTHS
    onset_types = ['early', 'late']
    
    n_rows = len(variables)
    n_cols = len(months) * len(onset_types)  # 4 columns
    
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(16, 14),
        subplot_kw={'projection': ccrs.PlateCarree()},
        constrained_layout=True
    )
    
    scale_factor = 1e-6  # J/m^2 to MJ/m^2
    
    # Store contour fills for colorbars (one per row)
    cf_per_row = {}
    levels_per_row = {}
    
    for row, var in enumerate(variables):
        # Determine color scale for this variable
        all_values = []
        for month in months:
            for onset_type in onset_types:
                if var in all_anomalies and month in all_anomalies[var]:
                    data = all_anomalies[var][month][onset_type]
                    all_values.extend(data.values.flatten()[~np.isnan(data.values.flatten())])
        
        if not all_values:
            continue
            
        vmax_data = np.percentile(np.abs(all_values), 98) * scale_factor
        vmax_scaled = np.ceil(vmax_data / interval) * interval
        if vmax_scaled == 0:
            vmax_scaled = interval
        vmin_scaled = -vmax_scaled
        levels = np.arange(vmin_scaled, vmax_scaled + interval, interval)
        levels_per_row[row] = levels
        
        col = 0
        for month in months:
            for onset_type in onset_types:
                ax = axes[row, col]
                
                if var not in all_anomalies or month not in all_anomalies[var]:
                    ax.set_visible(False)
                    col += 1
                    continue
                
                data = all_anomalies[var][month][onset_type] * scale_factor
                pval = all_pvalues[var][month][onset_type]
                
                # Get coordinates
                if 'longitude' in data.coords:
                    lon = data.longitude.values
                    lat = data.latitude.values
                else:
                    lon = data.lon.values
                    lat = data.lat.values
                
                # Subset to region
                lon_mask = (lon >= region[0]) & (lon <= region[1])
                lat_mask = (lat >= region[2]) & (lat <= region[3])
                
                lon_sub = lon[lon_mask]
                lat_sub = lat[lat_mask]
                data_sub = data.values[np.ix_(lat_mask, lon_mask)]
                pval_sub = pval.values[np.ix_(lat_mask, lon_mask)]
                
                # Plot
                cf = ax.contourf(
                    lon_sub, lat_sub, data_sub,
                    levels=levels,
                    cmap=cmap,
                    transform=ccrs.PlateCarree(),
                    extend='both'
                )
                
                # Store for colorbar (use last column)
                if col == n_cols - 1:
                    cf_per_row[row] = cf
                
                # Stippling
                sig_mask = pval_sub < alpha
                skip = 5
                lon_mesh, lat_mesh = np.meshgrid(lon_sub, lat_sub)
                ax.scatter(
                    lon_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                    lat_mesh[::skip, ::skip][sig_mask[::skip, ::skip]],
                    marker='.', s=0.3, c='black', alpha=0.5,
                    transform=ccrs.PlateCarree()
                )
                
                ax.coastlines(linewidth=0.4)
                ax.set_extent(region, crs=ccrs.PlateCarree())
                
                # Gridlines
                gl = ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4, linestyle='--')
                if col == 0:
                    gl.left_labels = True
                    gl.yformatter = LatitudeFormatter()
                if row == n_rows - 1:
                    gl.bottom_labels = True
                    gl.xformatter = LongitudeFormatter()
                
                # Column titles (top row only)
                if row == 0:
                    onset_label = 'Early' if onset_type == 'early' else 'Late'
                    ax.set_title(f"{MONTH_NAMES[month]}\n{onset_label}", fontsize=10, fontweight='bold')
                
                # Row labels (left side)
                if col == 0:
                    ax.text(-0.15, 0.5, VARIABLE_NAMES[var], transform=ax.transAxes,
                           fontsize=10, fontweight='bold', va='center', ha='right',
                           rotation=90)
                
                col += 1
    
    # Add colorbars on the right side for each row
    for row in range(n_rows):
        if row in cf_per_row:
            # Position colorbar to the right of each row
            cax = fig.add_axes([0.92, 0.78 - row*0.21, 0.012, 0.15])
            cbar = fig.colorbar(cf_per_row[row], cax=cax, orientation='vertical')
            cbar.set_label('MJ/m²', fontsize=8)
            cbar.ax.tick_params(labelsize=7)
    
    # Main title
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


def save_results_netcdf(all_anomalies: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                        all_pvalues: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                        all_climatologies: Dict[str, Dict[int, xr.DataArray]],
                        output_path: Path,
                        early_years: List[int],
                        late_years: List[int]) -> None:
    """
    Save analysis results to NetCDF file.
    """
    ds = xr.Dataset()
    
    for var in VARIABLES:
        if var not in all_anomalies:
            continue
            
        for month in ANALYSIS_MONTHS:
            if month not in all_anomalies[var]:
                continue
                
            month_name = MONTH_NAMES[month].lower()
            
            # Climatology
            if var in all_climatologies and month in all_climatologies[var]:
                ds[f'{var}_clim_{month_name}'] = all_climatologies[var][month]
                ds[f'{var}_clim_{month_name}'].attrs = {
                    'long_name': f'{VARIABLE_LONG_NAMES[var]} Climatology for {MONTH_NAMES[month]}',
                    'units': 'J m-2',
                    'period': f'{CLIM_START_YEAR}-{CLIM_END_YEAR}'
                }
            
            for onset_type in ['early', 'late']:
                years_list = early_years if onset_type == 'early' else late_years
                
                # Anomaly
                ds[f'{var}_anom_{month_name}_{onset_type}'] = all_anomalies[var][month][onset_type]
                ds[f'{var}_anom_{month_name}_{onset_type}'].attrs = {
                    'long_name': f'{VARIABLE_LONG_NAMES[var]} Anomaly for {MONTH_NAMES[month]} ({onset_type} onset)',
                    'units': 'J m-2',
                    'years': str(years_list)
                }
                
                # P-value
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
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze VIMSE and component anomalies for early/late monsoon onset years.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using directory of yearly VIMSE files
  python analyze_vimse_anomaly.py /path/to/vimse_data/ --output-dir ./output/

  # Using single combined file
  python analyze_vimse_anomaly.py /path/to/vimse_combined.nc --single-file --output-dir ./output/

  # Specify region for plotting
  python analyze_vimse_anomaly.py /path/to/data/ --output-dir ./output/ --region 60 120 -10 40

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
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for PDF plots (default: current directory)'
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
        '--region',
        type=float,
        nargs=4,
        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
        default=[45, 115, -10, 30],
        help='Plot region extent (default: 45 115 -10 30)'
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
        default=0.10,
        help='Significance level for t-test (default: 0.10 for 90%% confidence)'
    )
    
    parser.add_argument(
        '--interval',
        type=float,
        default=1.5,
        help='Colorbar interval in MJ/m² (default: 1.5)'
    )
    
    parser.add_argument(
        '--no-individual',
        action='store_true',
        help='Skip individual variable plots, only create combined plot'
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
    
    print("=" * 70)
    print("VIMSE and Component Anomaly Analysis for Monsoon Onset")
    print("=" * 70)
    print(f"Early onset years: {early_years}")
    print(f"Late onset years: {late_years}")
    print(f"Climatology period: {CLIM_START_YEAR}-{CLIM_END_YEAR}")
    print(f"Analysis months: {[MONTH_NAMES[m] for m in ANALYSIS_MONTHS]}")
    print(f"Significance level: {args.alpha} ({int((1-args.alpha)*100)}% confidence)")
    print(f"Variables: {VARIABLES}")
    print()
    
    # Load data
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
    
    # Detect time dimension
    sample_var = list(data_dict.values())[0]
    time_dim = 'valid_time' if 'valid_time' in sample_var.dims else 'time'
    print(f"Time dimension: {time_dim}")
    print()
    
    # Process each variable
    all_anomalies = {}
    all_pvalues = {}
    all_climatologies = {}
    
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
            
            # Extract month data
            data_month = extract_month_data(data, month, time_dim)
            
            # Calculate climatology
            climatology = calculate_climatology(data_month, CLIM_START_YEAR, CLIM_END_YEAR, time_dim)
            all_climatologies[var][month] = climatology
            
            all_anomalies[var][month] = {}
            all_pvalues[var][month] = {}
            
            # Calculate anomalies for early and late onset years
            for onset_type, years in [('early', early_years), ('late', late_years)]:
                # Calculate composite anomaly
                anomaly, composite_data = calculate_composite_anomaly(
                    data_month, climatology, years, time_dim
                )
                all_anomalies[var][month][onset_type] = anomaly
                
                # Perform t-test
                t_stat, pval = perform_ttest(composite_data, data_month, time_dim, args.alpha)
                all_pvalues[var][month][onset_type] = pval
                
                # Print statistics
                n_sig = (pval < args.alpha).sum().values
                total = pval.size
                print(f"    {onset_type.capitalize()}: {n_sig}/{total} significant ({100*n_sig/total:.1f}%)")
    
    print()
    
    # Create plots
    region = tuple(args.region)
    
    # Individual variable plots
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
    
    # Combined plot
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
    
    # Create separate 4x3 plots for early and late onset years (March, April, April-March)
    print("Creating 4x3 panel plots (March / April / April-March)...")
    
    # Early onset years
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
    
    # Late onset years
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
    
    # Save NetCDF results if requested
    if args.output_nc:
        print("Saving results to NetCDF...")
        save_results_netcdf(all_anomalies, all_pvalues, all_climatologies, 
                           Path(args.output_nc), early_years, late_years)
    
    print()
    print("=" * 70)
    print("Analysis complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    main()
