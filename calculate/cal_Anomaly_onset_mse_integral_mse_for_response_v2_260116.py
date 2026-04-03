"""
260116
Compute Moist Static Energy (MSE) and Vertically Integrated MSE (VIMSE) from ERA5 data.

MSE = cp*T + Phi + Lv*q
    where:
    - cp = 1004.0 J/kg/K (specific heat at constant pressure)
    - Lv = 2.5e6 J/kg (latent heat of vaporization)
    - T = temperature (K)
    - Phi = geopotential (m^2/s^2)
    - q = specific humidity (kg/kg)

VIMSE = (1/g) * integral(MSE * dp) from p_top to p_bottom
    where:
    - g = 9.80665 m/s^2
    - dp = pressure layer thickness (Pa)

Author: Generated script for ERA5 processing
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings

import numpy as np
import xarray as xr


# Physical constants
CP = 1004.0       # Specific heat at constant pressure [J/kg/K]
LV = 2.5e6        # Latent heat of vaporization [J/kg]
G = 9.80665       # Gravitational acceleration [m/s^2]

# Default pressure range for integration (hPa)
DEFAULT_P_TOP = 100.0     # Top of integration [hPa]
DEFAULT_P_BOTTOM = 1000.0  # Bottom of integration [hPa]


def parse_variable_mapping(mapping_str: Optional[str]) -> Dict[str, str]:
    """
    Parse variable name mapping from CLI string.
    
    Format: "t=temperature,q=humidity,z=geopotential,sp=surface_pressure"
    
    Parameters
    ----------
    mapping_str : str or None
        Comma-separated key=value pairs for variable mapping
        
    Returns
    -------
    dict
        Mapping from standard names (t, q, z, sp) to actual variable names in file
    """
    # Default ERA5 variable names
    default_mapping = {
        't': 't',       # Temperature
        'q': 'q',       # Specific humidity  
        'z': 'z',       # Geopotential
        'sp': 'sp',     # Surface pressure (optional)
    }
    
    if mapping_str is None:
        return default_mapping
    
    mapping = default_mapping.copy()
    for pair in mapping_str.split(','):
        pair = pair.strip()
        if '=' in pair:
            key, value = pair.split('=', 1)
            key = key.strip().lower()
            value = value.strip()
            if key in mapping:
                mapping[key] = value
            else:
                warnings.warn(f"Unknown variable key '{key}', ignoring.")
    
    return mapping


def compute_pressure_thickness(pressure_levels: np.ndarray, 
                               p_top: float = DEFAULT_P_TOP,
                               p_bottom: float = DEFAULT_P_BOTTOM) -> np.ndarray:
    """
    Compute pressure thickness (dp) for each level using layer boundaries.
    
    Uses the midpoint method: boundaries are placed halfway between levels.
    For edge levels, extends to p_top or p_bottom.
    
    Parameters
    ----------
    pressure_levels : np.ndarray
        Pressure levels in hPa (must be sorted, either increasing or decreasing)
    p_top : float
        Top pressure boundary in hPa
    p_bottom : float
        Bottom pressure boundary in hPa
        
    Returns
    -------
    np.ndarray
        Pressure thickness dp in Pa for each level
    """
    # Sort levels in decreasing order (high pressure to low pressure, surface to top)
    sorted_idx = np.argsort(pressure_levels)[::-1]
    p_sorted = pressure_levels[sorted_idx]
    
    # Filter levels within the integration range
    mask = (p_sorted >= p_top) & (p_sorted <= p_bottom)
    p_filtered = p_sorted[mask]
    
    if len(p_filtered) == 0:
        raise ValueError(f"No pressure levels found between {p_top} and {p_bottom} hPa")
    
    n_levels = len(p_filtered)
    dp = np.zeros(n_levels)
    
    # Compute layer boundaries (midpoints between levels)
    for i in range(n_levels):
        if i == 0:
            # Bottom level: boundary from p_bottom to midpoint with next level
            p_lower = min(p_bottom, p_filtered[i])
            if n_levels > 1:
                p_upper = 0.5 * (p_filtered[i] + p_filtered[i + 1])
            else:
                p_upper = max(p_top, p_filtered[i])
        elif i == n_levels - 1:
            # Top level: boundary from midpoint with previous level to p_top
            p_lower = 0.5 * (p_filtered[i - 1] + p_filtered[i])
            p_upper = max(p_top, p_filtered[i])
        else:
            # Interior level: midpoints on both sides
            p_lower = 0.5 * (p_filtered[i - 1] + p_filtered[i])
            p_upper = 0.5 * (p_filtered[i] + p_filtered[i + 1])
        
        dp[i] = (p_lower - p_upper) * 100.0  # Convert hPa to Pa
    
    # Create output array in original order
    dp_full = np.zeros(len(pressure_levels))
    filtered_indices = sorted_idx[mask]
    
    # Map back to original indices
    for i, idx in enumerate(filtered_indices):
        dp_full[idx] = dp[i]
    
    return dp_full


def compute_mse(ds: xr.Dataset, 
                var_mapping: Dict[str, str],
                level_dim: str = 'pressure_level') -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    Compute Moist Static Energy and its components.
    
    MSE = cp*T + Phi + Lv*q
    
    Parameters
    ----------
    ds : xr.Dataset
        Input dataset containing temperature, specific humidity, and geopotential
    var_mapping : dict
        Variable name mapping
    level_dim : str
        Name of the pressure level dimension
        
    Returns
    -------
    tuple
        (mse, cpt, phi, lvq) - MSE and its three components, all in J/kg
    """
    t_var = var_mapping['t']
    q_var = var_mapping['q']
    z_var = var_mapping['z']
    
    # Check required variables exist
    for name, var in [(t_var, 'temperature'), (q_var, 'specific humidity'), (z_var, 'geopotential')]:
        if name not in ds:
            raise KeyError(f"Variable '{name}' ({var}) not found in dataset. "
                          f"Available variables: {list(ds.data_vars)}")
    
    T = ds[t_var]  # Temperature [K]
    q = ds[q_var]  # Specific humidity [kg/kg]
    Phi = ds[z_var]  # Geopotential [m^2/s^2]
    
    # Compute MSE components
    cpt = CP * T      # Sensible heat term [J/kg]
    phi = Phi         # Geopotential term [J/kg] (already in correct units)
    lvq = LV * q      # Latent heat term [J/kg]
    
    # Compute total MSE
    mse = cpt + phi + lvq
    
    # Add metadata for MSE
    mse.name = 'mse'
    mse.attrs = {
        'long_name': 'Moist Static Energy',
        'units': 'J kg-1',
        'standard_name': 'moist_static_energy',
        'formula': 'MSE = cp*T + Phi + Lv*q',
        'cp': f'{CP} J/kg/K',
        'Lv': f'{LV} J/kg',
    }
    
    # Add metadata for cp*T (sensible heat term)
    cpt.name = 'cpt'
    cpt.attrs = {
        'long_name': 'Sensible Heat Term (cp*T)',
        'units': 'J kg-1',
        'formula': 'cp * T',
        'cp': f'{CP} J/kg/K',
    }
    
    # Add metadata for Phi (geopotential term)
    phi.name = 'phi'
    phi.attrs = {
        'long_name': 'Geopotential Term (Phi)',
        'units': 'J kg-1',
        'formula': 'Phi (geopotential)',
    }
    
    # Add metadata for Lv*q (latent heat term)
    lvq.name = 'lvq'
    lvq.attrs = {
        'long_name': 'Latent Heat Term (Lv*q)',
        'units': 'J kg-1',
        'formula': 'Lv * q',
        'Lv': f'{LV} J/kg',
    }
    
    return mse, cpt, phi, lvq


def compute_vertical_integral(field: xr.DataArray,
                              ds: xr.Dataset,
                              var_mapping: Dict[str, str],
                              level_dim: str = 'pressure_level',
                              p_top: float = DEFAULT_P_TOP,
                              p_bottom: float = DEFAULT_P_BOTTOM,
                              use_surface_pressure: bool = True,
                              field_name: str = 'vimse') -> Tuple[xr.DataArray, str]:
    """
    Compute vertically integrated field (mass-weighted column integral).
    
    VI_field = (1/g) * integral(field * dp)
    
    Parameters
    ----------
    field : xr.DataArray
        Field to integrate (e.g., MSE, cp*T, Phi, Lv*q)
    ds : xr.Dataset
        Original dataset (for surface pressure if available)
    var_mapping : dict
        Variable name mapping
    level_dim : str
        Name of pressure level dimension
    p_top : float
        Top of integration in hPa
    p_bottom : float
        Bottom of integration in hPa
    use_surface_pressure : bool
        Whether to use surface pressure for masking
    field_name : str
        Name for the output variable
        
    Returns
    -------
    tuple
        (vertically_integrated_field DataArray, integration_note string)
    """
    # Get pressure levels
    pressure_levels = field[level_dim].values
    
    # Compute dp for each level
    dp = compute_pressure_thickness(pressure_levels, p_top, p_bottom)
    
    # Create dp as DataArray for broadcasting
    dp_da = xr.DataArray(
        dp,
        dims=[level_dim],
        coords={level_dim: pressure_levels}
    )
    
    # Check for surface pressure
    sp_var = var_mapping.get('sp', 'sp')
    integration_note = ""
    
    if use_surface_pressure and sp_var in ds:
        # Use surface pressure to mask levels below surface
        sp = ds[sp_var]  # Surface pressure in Pa
        
        # Convert pressure levels to Pa for comparison
        p_levels_pa = pressure_levels * 100.0  # hPa to Pa
        p_levels_da = xr.DataArray(
            p_levels_pa,
            dims=[level_dim],
            coords={level_dim: pressure_levels}
        )
        
        # Create mask: True where pressure level is above (less than) surface pressure
        # We want to integrate where p <= ps
        mask = p_levels_da <= sp
        
        # Apply mask to field
        field_masked = field.where(mask, 0.0)
        
        # Also mask dp where level is below surface
        dp_masked = dp_da.where(mask, 0.0)
        
        integration_note = (f"Integrated from {p_top} hPa to surface pressure (sp). "
                          f"Levels where p > sp are masked.")
        
        # Compute vertical integral with masking
        vi_field = (1.0 / G) * (field_masked * dp_masked).sum(dim=level_dim)
        
    else:
        if use_surface_pressure:
            integration_note = (f"Surface pressure (sp) not available in dataset. "
                              f"Integrating over fixed range {p_top}-{p_bottom} hPa. "
                              f"This assumes all grid points have surface pressure >= {p_bottom} hPa.")
            warnings.warn(integration_note)
        else:
            integration_note = f"Integrated over fixed pressure range: {p_top}-{p_bottom} hPa"
        
        # Compute vertical integral without surface pressure masking
        vi_field = (1.0 / G) * (field * dp_da).sum(dim=level_dim, skipna=True)
    
    # Add metadata
    vi_field.name = field_name
    vi_field.attrs = {
        'long_name': f'Vertically Integrated {field.attrs.get("long_name", field_name)}',
        'units': 'J m-2',
        'formula': f'VI = (1/g) * integral({field.name} * dp)',
        'g': f'{G} m/s^2',
        'integration_range': f'{p_top}-{p_bottom} hPa',
        'integration_note': integration_note,
    }
    
    return vi_field, integration_note


def compute_vimse(mse: xr.DataArray,
                  ds: xr.Dataset,
                  var_mapping: Dict[str, str],
                  level_dim: str = 'pressure_level',
                  p_top: float = DEFAULT_P_TOP,
                  p_bottom: float = DEFAULT_P_BOTTOM,
                  use_surface_pressure: bool = True) -> Tuple[xr.DataArray, str]:
    """
    Compute vertically integrated MSE (wrapper for backward compatibility).
    
    VIMSE = (1/g) * integral(MSE * dp)
    
    Parameters
    ----------
    mse : xr.DataArray
        Moist Static Energy field
    ds : xr.Dataset
        Original dataset (for surface pressure if available)
    var_mapping : dict
        Variable name mapping
    level_dim : str
        Name of pressure level dimension
    p_top : float
        Top of integration in hPa
    p_bottom : float
        Bottom of integration in hPa
    use_surface_pressure : bool
        Whether to use surface pressure for masking
        
    Returns
    -------
    tuple
        (VIMSE DataArray, integration_note string)
    """
    return compute_vertical_integral(
        mse, ds, var_mapping, level_dim, p_top, p_bottom, 
        use_surface_pressure, field_name='vimse'
    )


def process_file(input_path: Path,
                 output_mse_path: Optional[Path],
                 output_vimse_path: Optional[Path],
                 var_mapping: Dict[str, str],
                 p_top: float = DEFAULT_P_TOP,
                 p_bottom: float = DEFAULT_P_BOTTOM,
                 use_surface_pressure: bool = True,
                 level_dim: str = 'pressure_level') -> None:
    """
    Process a single ERA5 file to compute MSE, VIMSE, and component terms.
    
    Parameters
    ----------
    input_path : Path
        Path to input NetCDF file
    output_mse_path : Path or None
        Path for MSE output file (includes MSE and components at pressure levels)
    output_vimse_path : Path or None
        Path for VIMSE output file (includes VIMSE and vertically integrated components)
    var_mapping : dict
        Variable name mapping
    p_top : float
        Top pressure for integration (hPa)
    p_bottom : float
        Bottom pressure for integration (hPa)
    use_surface_pressure : bool
        Whether to use surface pressure for masking
    level_dim : str
        Name of pressure level dimension
    """
    print(f"Processing: {input_path}")
    
    # Open dataset
    ds = xr.open_dataset(input_path)
    
    # Auto-detect level dimension if not found
    if level_dim not in ds.dims:
        possible_dims = ['pressure_level', 'level', 'plev', 'isobaricInhPa', 'lev']
        for dim in possible_dims:
            if dim in ds.dims:
                level_dim = dim
                print(f"  Using detected level dimension: {level_dim}")
                break
        else:
            raise ValueError(f"Could not find pressure level dimension. "
                           f"Available dimensions: {list(ds.dims)}")
    
    # Compute MSE and its components
    print("  Computing MSE and components...")
    mse, cpt, phi, lvq = compute_mse(ds, var_mapping, level_dim)
    
    # Compute vertically integrated fields
    print("  Computing vertically integrated MSE...")
    vimse, integration_note = compute_vertical_integral(
        mse, ds, var_mapping, level_dim, 
        p_top, p_bottom, use_surface_pressure, field_name='vimse'
    )
    print(f"  Integration note: {integration_note}")
    
    print("  Computing vertically integrated components...")
    vicpt, _ = compute_vertical_integral(
        cpt, ds, var_mapping, level_dim,
        p_top, p_bottom, use_surface_pressure, field_name='vicpt'
    )
    viphi, _ = compute_vertical_integral(
        phi, ds, var_mapping, level_dim,
        p_top, p_bottom, use_surface_pressure, field_name='viphi'
    )
    vilvq, _ = compute_vertical_integral(
        lvq, ds, var_mapping, level_dim,
        p_top, p_bottom, use_surface_pressure, field_name='vilvq'
    )
    
    # Update metadata for components
    vicpt.attrs['long_name'] = 'Vertically Integrated Sensible Heat Term (cp*T)'
    vicpt.attrs['standard_name'] = 'vertically_integrated_sensible_heat'
    
    viphi.attrs['long_name'] = 'Vertically Integrated Geopotential Term (Phi)'
    viphi.attrs['standard_name'] = 'vertically_integrated_geopotential'
    
    vilvq.attrs['long_name'] = 'Vertically Integrated Latent Heat Term (Lv*q)'
    vilvq.attrs['standard_name'] = 'vertically_integrated_latent_heat'
    
    # Global attributes for outputs
    global_attrs = {
        'title': 'Moist Static Energy computed from ERA5',
        'source': str(input_path),
        'institution': 'Computed from ERA5 reanalysis data',
        'references': 'ERA5: Hersbach et al. (2020)',
        'Conventions': 'CF-1.7',
        'history': f'Created by compute_mse.py',
        'cp': f'{CP} J/kg/K (specific heat at constant pressure)',
        'Lv': f'{LV} J/kg (latent heat of vaporization)',
        'g': f'{G} m/s^2 (gravitational acceleration)',
    }
    
    # Save MSE and components at pressure levels
    if output_mse_path:
        print(f"  Saving MSE and components to: {output_mse_path}")
        mse_ds = xr.Dataset({
            'mse': mse,
            'cpt': cpt,
            'phi': phi,
            'lvq': lvq,
        })
        mse_ds.attrs = global_attrs
        mse_ds.attrs['description'] = ('Moist Static Energy and components: '
                                      'MSE = cp*T + Phi + Lv*q')
        
        # Set encoding for efficient storage
        encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 4}
                   for var in ['mse', 'cpt', 'phi', 'lvq']}
        mse_ds.to_netcdf(output_mse_path, encoding=encoding)
    
    # Save VIMSE and vertically integrated components
    if output_vimse_path:
        print(f"  Saving VIMSE and components to: {output_vimse_path}")
        vimse_ds = xr.Dataset({
            'vimse': vimse,
            'vicpt': vicpt,
            'viphi': viphi,
            'vilvq': vilvq,
        })
        vimse_ds.attrs = global_attrs
        vimse_ds.attrs['description'] = ('Vertically Integrated Moist Static Energy and components: '
                                        'VIMSE = VI(cp*T) + VI(Phi) + VI(Lv*q)')
        vimse_ds.attrs['integration_note'] = integration_note
        
        encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 4}
                   for var in ['vimse', 'vicpt', 'viphi', 'vilvq']}
        vimse_ds.to_netcdf(output_vimse_path, encoding=encoding)
    
    ds.close()
    print("  Done!")


def process_directory(input_dir: Path,
                      output_dir: Path,
                      var_mapping: Dict[str, str],
                      p_top: float = DEFAULT_P_TOP,
                      p_bottom: float = DEFAULT_P_BOTTOM,
                      use_surface_pressure: bool = True,
                      file_pattern: str = "*.nc") -> None:
    """
    Process all NetCDF files in a directory.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing input NetCDF files
    output_dir : Path
        Directory for output files
    var_mapping : dict
        Variable name mapping
    p_top : float
        Top pressure for integration (hPa)
    p_bottom : float
        Bottom pressure for integration (hPa)
    use_surface_pressure : bool
        Whether to use surface pressure for masking
    file_pattern : str
        Glob pattern for input files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    mse_dir = output_dir / 'mse'
    vimse_dir = output_dir / 'vimse'
    mse_dir.mkdir(parents=True, exist_ok=True)
    vimse_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    input_files = sorted(input_dir.glob(file_pattern))
    
    if not input_files:
        raise FileNotFoundError(f"No files matching '{file_pattern}' found in {input_dir}")
    
    print(f"Found {len(input_files)} files to process")
    
    for input_file in input_files:
        # Generate output filenames
        stem = input_file.stem
        mse_output = mse_dir / f"{stem}_mse.nc"
        vimse_output = vimse_dir / f"{stem}_vimse.nc"
        
        try:
            process_file(
                input_file, mse_output, vimse_output,
                var_mapping, p_top, p_bottom, use_surface_pressure
            )
        except Exception as e:
            print(f"  ERROR processing {input_file}: {e}")
            continue
    
    print(f"\nProcessing complete!")
    print(f"MSE files saved to: {mse_dir}")
    print(f"VIMSE files saved to: {vimse_dir}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Compute Moist Static Energy (MSE) and Vertically Integrated MSE (VIMSE) from ERA5 data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python compute_mse.py input.nc --output-mse mse.nc --output-vimse vimse.nc

  # Process all files in a directory
  python compute_mse.py /data/ERA5/ --output-dir /output/ --batch

  # Custom pressure range
  python compute_mse.py input.nc --output-mse mse.nc --p-top 200 --p-bottom 850

  # Custom variable names
  python compute_mse.py input.nc --output-mse mse.nc --var-map "t=temp,q=shum,z=geopot"

Physical constants used:
  cp = 1004.0 J/kg/K  (specific heat at constant pressure)
  Lv = 2.5e6 J/kg     (latent heat of vaporization)
  g  = 9.80665 m/s^2  (gravitational acceleration)
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input NetCDF file or directory (with --batch)'
    )
    
    parser.add_argument(
        '--output-mse',
        type=str,
        default=None,
        help='Output path for MSE NetCDF file (single file mode)'
    )
    
    parser.add_argument(
        '--output-vimse',
        type=str,
        default=None,
        help='Output path for VIMSE NetCDF file (single file mode)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for batch processing (creates mse/ and vimse/ subdirs)'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all NetCDF files in input directory'
    )
    
    parser.add_argument(
        '--p-top',
        type=float,
        default=DEFAULT_P_TOP,
        help=f'Top pressure level for integration in hPa (default: {DEFAULT_P_TOP})'
    )
    
    parser.add_argument(
        '--p-bottom',
        type=float,
        default=DEFAULT_P_BOTTOM,
        help=f'Bottom pressure level for integration in hPa (default: {DEFAULT_P_BOTTOM})'
    )
    
    parser.add_argument(
        '--var-map',
        type=str,
        default=None,
        help='Variable name mapping: "t=temp,q=shum,z=geopot,sp=ps" (default: t,q,z,sp)'
    )
    
    parser.add_argument(
        '--no-sp-mask',
        action='store_true',
        help='Disable surface pressure masking (integrate over fixed level range)'
    )
    
    parser.add_argument(
        '--file-pattern',
        type=str,
        default='*.nc',
        help='File pattern for batch mode (default: "*.nc")'
    )
    
    parser.add_argument(
        '--level-dim',
        type=str,
        default='pressure_level',
        help='Name of pressure level dimension (default: pressure_level)'
    )
    
    args = parser.parse_args()
    
    # Parse variable mapping
    var_mapping = parse_variable_mapping(args.var_map)
    print(f"Variable mapping: {var_mapping}")
    print(f"Integration range: {args.p_top} - {args.p_bottom} hPa")
    print(f"Surface pressure masking: {'disabled' if args.no_sp_mask else 'enabled (if sp available)'}")
    print()
    
    input_path = Path(args.input)
    
    if args.batch:
        # Batch processing mode
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory. Use --batch with a directory.", 
                  file=sys.stderr)
            sys.exit(1)
        
        if args.output_dir is None:
            print("Error: --output-dir required for batch mode.", file=sys.stderr)
            sys.exit(1)
        
        process_directory(
            input_path,
            Path(args.output_dir),
            var_mapping,
            args.p_top,
            args.p_bottom,
            not args.no_sp_mask,
            args.file_pattern
        )
    else:
        # Single file mode
        if not input_path.is_file():
            print(f"Error: {input_path} is not a file.", file=sys.stderr)
            sys.exit(1)
        
        if args.output_mse is None and args.output_vimse is None:
            print("Error: At least one of --output-mse or --output-vimse required.", 
                  file=sys.stderr)
            sys.exit(1)
        
        output_mse = Path(args.output_mse) if args.output_mse else None
        output_vimse = Path(args.output_vimse) if args.output_vimse else None
        
        process_file(
            input_path,
            output_mse,
            output_vimse,
            var_mapping,
            args.p_top,
            args.p_bottom,
            not args.no_sp_mask,
            args.level_dim
        )


if __name__ == '__main__':
    main()
