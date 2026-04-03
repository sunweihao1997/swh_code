#!/usr/bin/env python3
"""
Plot meridional profile of column-integrated Lv*q (VILVQ) for early and late monsoon onset years.
Ocean-only version: masks land points using ERA5 land-sea mask.

This script:
1. Reads VILVQ data from NetCDF files
2. Applies land-sea mask (ocean only, land = NaN)
3. Averages over 85-100°E longitude (skipping NaN)
4. Creates meridional profiles (10°S - 25°N) 
5. Plots February, March, April for early and late onset composites
6. Computes d(Lvq)/dy derivative for early-late difference

Output: 3×3 panel figure
- Row 1: Early onset years (Feb, Mar, Apr) - VI(Lvq) profiles with climatology
- Row 2: Late onset years (Feb, Mar, Apr) - VI(Lvq) profiles with climatology
- Row 3: Early - Late difference (Feb, Mar, Apr) - d(VI(Lvq))/dy derivative
        where y = a·φ, φ in radians, a = 6.371×10⁶ m (Earth's radius)

Author: Generated script for ERA5 VILVQ meridional profile analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# Define early and late onset years
EARLY_ONSET_YEARS = [1984, 1985, 1999, 2000, 2009, 2017]
LATE_ONSET_YEARS = [1983, 1987, 1993, 1997, 2010, 2016, 2018, 2020, 2021]

# Analysis period
CLIM_START_YEAR = 1980
CLIM_END_YEAR = 2021

# Months to analyze (February=2, March=3, April=4)
ANALYSIS_MONTHS = [2, 3, 4]
MONTH_NAMES = {2: 'February', 3: 'March', 4: 'April'}

# Region for averaging
LON_MIN = 85.0
LON_MAX = 100.0
LAT_MIN = -10.0
LAT_MAX = 25.0

# Land-sea mask threshold (values > threshold are land)
LSM_THRESHOLD = 0.5

# Earth's radius in meters
EARTH_RADIUS = 6.371e6  # m


def compute_meridional_derivative(data: xr.DataArray,
                                  lat_dim: str = 'latitude') -> xr.DataArray:
    """
    Compute derivative with respect to meridional distance y = a * phi.
    
    d(field)/dy = (1/a) * d(field)/d(phi)
    
    where phi is latitude in radians and a is Earth's radius.
    
    Parameters
    ----------
    data : xr.DataArray
        Input data with latitude dimension
    lat_dim : str
        Name of latitude dimension
        
    Returns
    -------
    xr.DataArray
        Derivative d(data)/dy in units of [data_units]/m
    """
    # Get latitude in degrees
    lat_deg = data[lat_dim].values
    
    # Convert to radians
    lat_rad = np.deg2rad(lat_deg)
    
    # Compute d(phi) in radians (spacing between latitude points)
    dphi = np.gradient(lat_rad)
    
    # Compute dy = a * dphi (meridional distance in meters)
    dy = EARTH_RADIUS * dphi
    
    # Compute derivative using numpy gradient
    # gradient returns derivative with respect to the coordinate spacing
    data_values = data.values
    
    # Find the axis corresponding to latitude
    lat_axis = data.dims.index(lat_dim)
    
    # Compute gradient along latitude axis
    ddata_dphi = np.gradient(data_values, lat_rad, axis=lat_axis)
    
    # Convert to derivative with respect to y: d/dy = (1/a) * d/dphi
    ddata_dy = ddata_dphi / EARTH_RADIUS
    
    # Create DataArray with same coordinates
    result = xr.DataArray(
        ddata_dy,
        coords=data.coords,
        dims=data.dims,
        attrs={
            'long_name': f'd({data.name})/dy',
            'units': 'J m-3' if 'J m-2' in str(data.attrs.get('units', '')) else f'{data.attrs.get("units", "")}/m'
        }
    )
    
    return result


def load_land_sea_mask(lsm_path: Path, 
                       lsm_var: str = 'lsm') -> xr.DataArray:
    """
    Load ERA5 land-sea mask.
    
    Parameters
    ----------
    lsm_path : Path
        Path to land-sea mask NetCDF file
    lsm_var : str
        Variable name for land-sea mask
        
    Returns
    -------
    xr.DataArray
        Land-sea mask (0 = ocean, 1 = land)
    """
    ds = xr.open_dataset(lsm_path)
    
    # Try to find the lsm variable
    if lsm_var in ds:
        lsm = ds[lsm_var]
    else:
        # Try common alternative names
        alt_names = ['lsm', 'LSM', 'land_sea_mask', 'mask', 'LAND_SEA_MASK']
        for name in alt_names:
            if name in ds:
                lsm = ds[name]
                print(f"  Found land-sea mask variable: {name}")
                break
        else:
            # Use first variable
            var_name = list(ds.data_vars)[0]
            lsm = ds[var_name]
            print(f"  Using variable '{var_name}' as land-sea mask")
    
    # If lsm has time dimension, take first time step
    if 'time' in lsm.dims:
        lsm = lsm.isel(time=0)
    if 'valid_time' in lsm.dims:
        lsm = lsm.isel(valid_time=0)
    
    # Squeeze any singleton dimensions
    lsm = lsm.squeeze()
    
    ds.close()
    
    return lsm


def create_ocean_mask(lsm: xr.DataArray, 
                      threshold: float = LSM_THRESHOLD) -> xr.DataArray:
    """
    Create ocean mask from land-sea mask.
    
    Parameters
    ----------
    lsm : xr.DataArray
        Land-sea mask (0 = ocean, 1 = land, or fractional)
    threshold : float
        Threshold for land (values > threshold are land)
        
    Returns
    -------
    xr.DataArray
        Ocean mask (True = ocean, False = land)
    """
    # Create ocean mask (True where ocean)
    ocean_mask = lsm <= threshold
    
    return ocean_mask


def regrid_mask_to_data(mask: xr.DataArray, 
                        data: xr.DataArray) -> xr.DataArray:
    """
    Regrid mask to match data grid if necessary.
    
    Parameters
    ----------
    mask : xr.DataArray
        Mask to regrid
    data : xr.DataArray
        Target data grid
        
    Returns
    -------
    xr.DataArray
        Regridded mask
    """
    # Get dimension names
    mask_lat_dim = 'latitude' if 'latitude' in mask.dims else 'lat'
    mask_lon_dim = 'longitude' if 'longitude' in mask.dims else 'lon'
    data_lat_dim = 'latitude' if 'latitude' in data.dims else 'lat'
    data_lon_dim = 'longitude' if 'longitude' in data.dims else 'lon'
    
    mask_lats = mask[mask_lat_dim].values
    mask_lons = mask[mask_lon_dim].values
    data_lats = data[data_lat_dim].values
    data_lons = data[data_lon_dim].values
    
    # Check if grids match
    if (len(mask_lats) == len(data_lats) and 
        len(mask_lons) == len(data_lons) and
        np.allclose(mask_lats, data_lats, atol=0.01) and
        np.allclose(mask_lons, data_lons, atol=0.01)):
        # Grids match, just rename dimensions if needed
        if mask_lat_dim != data_lat_dim or mask_lon_dim != data_lon_dim:
            mask = mask.rename({mask_lat_dim: data_lat_dim, mask_lon_dim: data_lon_dim})
        return mask
    
    # Need to interpolate
    print(f"  Regridding mask from ({len(mask_lats)}x{len(mask_lons)}) to ({len(data_lats)}x{len(data_lons)})")
    
    # Use nearest neighbor interpolation for mask
    mask_regridded = mask.interp(
        {mask_lat_dim: data_lats, mask_lon_dim: data_lons},
        method='nearest'
    )
    
    # Rename dimensions to match data
    if mask_lat_dim != data_lat_dim:
        mask_regridded = mask_regridded.rename({mask_lat_dim: data_lat_dim})
    if mask_lon_dim != data_lon_dim:
        mask_regridded = mask_regridded.rename({mask_lon_dim: data_lon_dim})
    
    return mask_regridded


def load_vilvq_data(data_path: Path, 
                    file_pattern: str = "ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc",
                    start_year: int = CLIM_START_YEAR,
                    end_year: int = CLIM_END_YEAR,
                    var_name: str = 'vilvq') -> xr.DataArray:
    """
    Load VILVQ data from multiple yearly files.
    
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
    var_name : str
        Variable name to load
        
    Returns
    -------
    xr.DataArray
        Combined VILVQ data with time dimension
    """
    datasets = []
    
    for year in range(start_year, end_year + 1):
        filename = file_pattern.format(year=year)
        filepath = data_path / filename
        
        if not filepath.exists():
            warnings.warn(f"File not found for year {year}: {filepath}")
            continue
        
        try:
            ds = xr.open_dataset(filepath)
            if var_name in ds:
                datasets.append(ds[var_name])
            else:
                data_vars = list(ds.data_vars)
                warnings.warn(f"Variable '{var_name}' not found in {filepath}. "
                            f"Available: {data_vars}")
            ds.close()
        except Exception as e:
            warnings.warn(f"Error loading {filepath}: {e}")
            continue
    
    if not datasets:
        raise FileNotFoundError(f"No VILVQ data files found in {data_path}")
    
    # Concatenate along time dimension
    vilvq = xr.concat(datasets, dim='valid_time')
    
    return vilvq


def apply_ocean_mask(data: xr.DataArray, 
                     ocean_mask: xr.DataArray) -> xr.DataArray:
    """
    Apply ocean mask to data (set land to NaN).
    
    Parameters
    ----------
    data : xr.DataArray
        Input data
    ocean_mask : xr.DataArray
        Ocean mask (True = ocean, False = land)
        
    Returns
    -------
    xr.DataArray
        Data with land points set to NaN
    """
    # Apply mask: where ocean_mask is False (land), set to NaN
    masked_data = data.where(ocean_mask)
    
    return masked_data


def extract_month_data(data: xr.DataArray, 
                       month: int,
                       time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Extract data for a specific month across all years.
    """
    time_coord = data[time_dim]
    
    if hasattr(time_coord.dt, 'month'):
        month_mask = time_coord.dt.month == month
    else:
        time_coord = xr.DataArray(
            np.array(time_coord.values, dtype='datetime64[ns]'),
            dims=[time_dim]
        )
        month_mask = time_coord.dt.month == month
    
    data_month = data.sel({time_dim: month_mask})
    
    return data_month


def compute_zonal_mean_ocean(data: xr.DataArray,
                             lon_min: float = LON_MIN,
                             lon_max: float = LON_MAX,
                             lat_min: float = LAT_MIN,
                             lat_max: float = LAT_MAX) -> xr.DataArray:
    """
    Compute zonal mean over specified longitude range, skipping NaN (land) values.
    
    Parameters
    ----------
    data : xr.DataArray
        Input data with longitude and latitude dimensions (land already masked as NaN)
    lon_min, lon_max : float
        Longitude range for averaging
    lat_min, lat_max : float
        Latitude range for subsetting
        
    Returns
    -------
    xr.DataArray
        Zonal mean as function of latitude (ocean-only average)
    """
    # Detect dimension names
    lon_dim = 'longitude' if 'longitude' in data.dims else 'lon'
    lat_dim = 'latitude' if 'latitude' in data.dims else 'lat'
    
    # Get coordinate values
    lons = data[lon_dim].values
    lats = data[lat_dim].values
    
    # Create masks
    lon_mask = (lons >= lon_min) & (lons <= lon_max)
    lat_mask = (lats >= lat_min) & (lats <= lat_max)
    
    # Subset data
    data_subset = data.isel({lon_dim: lon_mask, lat_dim: lat_mask})
    
    # Compute zonal mean, skipping NaN values
    zonal_mean = data_subset.mean(dim=lon_dim, skipna=True)
    
    return zonal_mean


def compute_climatology(data_month: xr.DataArray,
                        start_year: int = CLIM_START_YEAR,
                        end_year: int = CLIM_END_YEAR,
                        time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Calculate climatological mean for a specific month.
    """
    years = data_month[time_dim].dt.year
    year_mask = (years >= start_year) & (years <= end_year)
    data_clim_period = data_month.sel({time_dim: year_mask})
    
    climatology = data_clim_period.mean(dim=time_dim, skipna=True)
    
    return climatology


def compute_composite(data_month: xr.DataArray,
                      years: List[int],
                      time_dim: str = 'valid_time') -> xr.DataArray:
    """
    Calculate composite mean for selected years.
    """
    year_values = data_month[time_dim].dt.year.values
    year_mask = np.isin(year_values, years)
    
    if not np.any(year_mask):
        raise ValueError(f"No data found for years: {years}")
    
    data_composite = data_month.isel({time_dim: year_mask})
    composite_mean = data_composite.mean(dim=time_dim, skipna=True)
    
    return composite_mean


def plot_meridional_profiles_with_anomaly(profiles: Dict[str, Dict[int, Dict[str, xr.DataArray]]],
                                          output_path: Path,
                                          early_years: List[int],
                                          late_years: List[int],
                                          lon_range: Tuple[float, float] = (LON_MIN, LON_MAX),
                                          lat_range: Tuple[float, float] = (LAT_MIN, LAT_MAX),
                                          figsize: Tuple[int, int] = (14, 12)) -> None:
    """
    Create 3x3 panel plot showing composite, climatology, anomaly, and derivative.
    
    Layout:
    - Row 1: Early onset years (Feb, Mar, Apr) - VI(Lvq) profiles
    - Row 2: Late onset years (Feb, Mar, Apr) - VI(Lvq) profiles
    - Row 3: Early - Late difference (Feb, Mar, Apr) - d(VI(Lvq))/dy derivative
    
    Each panel in rows 1-2: composite (solid) + climatology (dashed)
    Row 3: derivative of (Early - Late) composite difference
    """
    fig, axes = plt.subplots(3, 3, figsize=figsize, constrained_layout=True)
    
    onset_types = ['early', 'late']
    months = ANALYSIS_MONTHS
    
    # Convert to MJ/m^2
    scale_factor = 1e-6
    
    # Colors
    color_composite = '#d62728'  # Red for composite
    color_clim = '#1f77b4'  # Blue for climatology
    color_fill_pos = '#ff9999'  # Light red for positive anomaly
    color_fill_neg = '#9999ff'  # Light blue for negative anomaly
    color_diff = '#2ca02c'  # Green for difference
    color_deriv = '#9467bd'  # Purple for derivative
    
    # ===== Rows 1-2: Early and Late onset profiles =====
    for i, onset_type in enumerate(onset_types):
        for j, month in enumerate(months):
            ax = axes[i, j]
            
            composite = profiles[onset_type][month]['composite'] * scale_factor
            clim = profiles[onset_type][month]['climatology'] * scale_factor
            
            # Get latitude coordinate
            lat_dim = 'latitude' if 'latitude' in composite.dims else 'lat'
            lats = composite[lat_dim].values
            
            comp_vals = composite.values
            clim_vals = clim.values
            
            # Plot climatology
            ax.plot(lats, clim_vals, 
                   color=color_clim, 
                   linestyle='--', 
                   linewidth=2,
                   label='Climatology')
            
            # Plot composite
            onset_label = 'Early' if onset_type == 'early' else 'Late'
            ax.plot(lats, comp_vals, 
                   color=color_composite, 
                   linestyle='-', 
                   linewidth=2.5,
                   label=f'{onset_label} Composite')
            
            # Fill between with different colors for positive/negative anomaly
            ax.fill_between(lats, clim_vals, comp_vals, 
                           where=(comp_vals >= clim_vals),
                           alpha=0.4, color=color_fill_pos,
                           interpolate=True)
            ax.fill_between(lats, clim_vals, comp_vals, 
                           where=(comp_vals < clim_vals),
                           alpha=0.4, color=color_fill_neg,
                           interpolate=True)
            
            # Formatting
            ax.set_xlim(lat_range)
            if j == 0:
                ax.set_ylabel('VI(Lv×q) (MJ/m²)', fontsize=10)
            ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
            # Grid
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # Title (only for first row)
            if i == 0:
                ax.set_title(f'{MONTH_NAMES[month]}', fontsize=12, fontweight='bold')
            
            # Legend
            if j == 2:
                ax.legend(loc='upper right', fontsize=8)
    
    # ===== Row 3: Early - Late difference derivative =====
    for j, month in enumerate(months):
        ax = axes[2, j]
        
        # Get latitude coordinate
        early_comp = profiles['early'][month]['composite']
        late_comp = profiles['late'][month]['composite']
        lat_dim = 'latitude' if 'latitude' in early_comp.dims else 'lat'
        lats = early_comp[lat_dim].values
        
        # Compute Early - Late difference
        diff = early_comp - late_comp
        
        # Compute derivative d(diff)/dy
        diff_derivative = compute_meridional_derivative(diff, lat_dim)
        
        # Scale: convert J/m^2 to MJ/m^2, then /m for derivative
        # Result is in MJ/m^3, but values will be very small
        # Better to use units of MJ/m^2 per 1000km = MJ/m^2 / (10^6 m)
        deriv_scale = 1e-6 * 1e6  # MJ/m^2 per 1000km (= 1 for J/m^3)
        deriv_vals = diff_derivative.values * deriv_scale
        
        # Also plot the difference itself (scaled)
        diff_vals = diff.values * scale_factor
        
        # Create twin axis for difference
        ax2 = ax.twinx()
        
        # Plot derivative on primary axis
        ax.plot(lats, deriv_vals, 
               color=color_deriv, 
               linestyle='-', 
               linewidth=2.5,
               label=r'd(VI(Lv×q))/dy')
        ax.fill_between(lats, 0, deriv_vals,
                       where=(deriv_vals >= 0),
                       alpha=0.3, color='#d4b4e2',
                       interpolate=True)
        ax.fill_between(lats, 0, deriv_vals,
                       where=(deriv_vals < 0),
                       alpha=0.3, color='#b4e2d4',
                       interpolate=True)
        
        # Plot difference on secondary axis (dashed)
        ax2.plot(lats, diff_vals,
                color=color_diff,
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label='Early−Late')
        
        # Formatting for primary axis (derivative)
        ax.set_xlim(lat_range)
        ax.set_xlabel('Latitude (°N)', fontsize=10)
        if j == 0:
            ax.set_ylabel(r'd(VI(Lv×q))/dy (MJ m$^{-2}$ / 1000km)', fontsize=9, color=color_deriv)
        ax.tick_params(axis='y', labelcolor=color_deriv)
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Formatting for secondary axis (difference)
        if j == 2:
            ax2.set_ylabel('Early−Late (MJ/m²)', fontsize=9, color=color_diff)
        ax2.tick_params(axis='y', labelcolor=color_diff)
        
        # Legend
        if j == 2:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    
    # Row labels
    row_labels = ['Early Onset', 'Late Onset', 'Early − Late\n(Derivative)']
    for i, label in enumerate(row_labels):
        if i < 2:
            years = early_years if i == 0 else late_years
            full_label = f'{label}\n({len(years)} years)'
        else:
            full_label = label
        axes[i, 0].annotate(full_label, xy=(-0.3, 0.5), xycoords='axes fraction',
                           fontsize=10, fontweight='bold', va='center', ha='center',
                           rotation=90)
    
    # Main title
#    fig.suptitle(
#        f'Meridional Profile of Vertically Integrated Latent Heat (Lv×q)\n'
#        f'Averaged over {lon_range[0]}°E–{lon_range[1]}°E (Ocean Only)\n'
#        f'Row 3: d(Early−Late)/dy where y = a·φ (a = 6.371×10⁶ m)',
#        fontsize=12, fontweight='bold', y=1.02
#    )
    
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Figure saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Plot meridional profile of VI(Lvq) for early/late monsoon onset (ocean only).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_meridional_vilvq_ocean.py /path/to/vimse_data/ \\
      --lsm-file /path/to/ERA5_lsm.nc \\
      --output meridional_vilvq_ocean.pdf

  # Custom longitude range
  python plot_meridional_vilvq_ocean.py /path/to/data/ \\
      --lsm-file lsm.nc --output plot.pdf --lon-range 80 105

Default settings:
  Longitude average: 85-100°E
  Latitude range: 10°S-25°N
  Months: February, March, April
  Early years: {early}
  Late years: {late}
        """.format(early=EARLY_ONSET_YEARS, late=LATE_ONSET_YEARS)
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input path: directory with yearly VIMSE files'
    )
    
    parser.add_argument(
        '--lsm-file',
        type=str,
        required=True,
        help='Path to ERA5 land-sea mask NetCDF file (required)'
    )
    
    parser.add_argument(
        '--lsm-var',
        type=str,
        default='lsm',
        help='Variable name for land-sea mask (default: lsm)'
    )
    
    parser.add_argument(
        '--lsm-threshold',
        type=float,
        default=0.5,
        help='Land-sea mask threshold (values > threshold = land, default: 0.5)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='meridional_vilvq_ocean.pdf',
        help='Output PDF path (default: meridional_vilvq_ocean.pdf)'
    )
    
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='ERA5_monnthly_pressure.0.5x0.5.{year}_vimse.nc',
        help='File pattern with {year} placeholder'
    )
    
    parser.add_argument(
        '--var-name',
        type=str,
        default='vilvq',
        help='Variable name for VI(Lvq) (default: vilvq)'
    )
    
    parser.add_argument(
        '--lon-range',
        type=float,
        nargs=2,
        metavar=('LON_MIN', 'LON_MAX'),
        default=[85, 100],
        help='Longitude range for averaging (default: 85 100)'
    )
    
    parser.add_argument(
        '--lat-range',
        type=float,
        nargs=2,
        metavar=('LAT_MIN', 'LAT_MAX'),
        default=[-10, 25],
        help='Latitude range for plotting (default: -10 25)'
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
    
    lon_range = tuple(args.lon_range)
    lat_range = tuple(args.lat_range)
    
    print("=" * 70)
    print("Meridional Profile of VI(Lv×q) for Monsoon Onset (Ocean Only)")
    print("=" * 70)
    print(f"Early onset years: {early_years}")
    print(f"Late onset years: {late_years}")
    print(f"Climatology period: {CLIM_START_YEAR}-{CLIM_END_YEAR}")
    print(f"Analysis months: {[MONTH_NAMES[m] for m in ANALYSIS_MONTHS]}")
    print(f"Longitude range for averaging: {lon_range[0]}°E - {lon_range[1]}°E")
    print(f"Latitude range for plotting: {lat_range[0]}°N - {lat_range[1]}°N")
    print(f"Land-sea mask file: {args.lsm_file}")
    print(f"Land-sea mask threshold: {args.lsm_threshold}")
    print()
    
    # Load land-sea mask
    print("Loading land-sea mask...")
    lsm_path = Path(args.lsm_file)
    if not lsm_path.exists():
        raise FileNotFoundError(f"Land-sea mask file not found: {lsm_path}")
    
    lsm = load_land_sea_mask(lsm_path, args.lsm_var)
    ocean_mask = create_ocean_mask(lsm, args.lsm_threshold)
    print(f"  Land-sea mask shape: {lsm.shape}")
    print(f"  Ocean fraction: {float(ocean_mask.mean()):.2%}")
    print()
    
    # Load data
    input_path = Path(args.input)
    print(f"Loading VI(Lv×q) data from: {input_path}")
    
    vilvq = load_vilvq_data(input_path, args.file_pattern, 
                            CLIM_START_YEAR, CLIM_END_YEAR, args.var_name)
    
    # Detect time dimension
    time_dim = 'valid_time' if 'valid_time' in vilvq.dims else 'time'
    print(f"Time dimension: {time_dim}")
    print(f"Data shape: {vilvq.shape}")
    print()
    
    # Regrid mask to data grid if necessary
    print("Checking grid compatibility...")
    ocean_mask_regridded = regrid_mask_to_data(ocean_mask, vilvq.isel({time_dim: 0}))
    print()
    
    # Apply ocean mask to data
    print("Applying ocean mask (setting land to NaN)...")
    vilvq_ocean = apply_ocean_mask(vilvq, ocean_mask_regridded)
    
    # Count ocean vs land points in the averaging region
    lon_dim = 'longitude' if 'longitude' in vilvq.dims else 'lon'
    lat_dim = 'latitude' if 'latitude' in vilvq.dims else 'lat'
    lons = vilvq[lon_dim].values
    lats = vilvq[lat_dim].values
    lon_mask_idx = (lons >= lon_range[0]) & (lons <= lon_range[1])
    lat_mask_idx = (lats >= lat_range[0]) & (lats <= lat_range[1])
    region_mask = ocean_mask_regridded.isel({lon_dim: lon_mask_idx, lat_dim: lat_mask_idx})
    ocean_points = int(region_mask.sum())
    total_points = int(region_mask.size)
    print(f"  In averaging region: {ocean_points}/{total_points} ocean points ({100*ocean_points/total_points:.1f}%)")
    print()
    
    # Process each month
    profiles = {
        'early': {},
        'late': {}
    }
    
    for month in ANALYSIS_MONTHS:
        print(f"Processing {MONTH_NAMES[month]}...")
        
        # Extract month data
        vilvq_month = extract_month_data(vilvq_ocean, month, time_dim)
        print(f"  Found {vilvq_month[time_dim].size} time steps")
        
        # Compute zonal mean (average over longitude, skipping NaN)
        vilvq_zonal = compute_zonal_mean_ocean(vilvq_month, 
                                               lon_range[0], lon_range[1],
                                               lat_range[0], lat_range[1])
        
        # Compute climatology
        clim = compute_climatology(vilvq_zonal, CLIM_START_YEAR, CLIM_END_YEAR, time_dim)
        
        # Compute composites
        for onset_type, years in [('early', early_years), ('late', late_years)]:
            composite = compute_composite(vilvq_zonal, years, time_dim)
            
            profiles[onset_type][month] = {
                'composite': composite,
                'climatology': clim
            }
            
            print(f"  {onset_type.capitalize()} composite computed")
    
    print()
    
    # Create plot
    print("Creating meridional profile plot...")
    output_path = Path(args.output)
    
    plot_meridional_profiles_with_anomaly(
        profiles, output_path,
        early_years, late_years,
        lon_range, lat_range
    )
    
    print()
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
