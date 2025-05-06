'''
2023-11-6
This script is to extract prect data and mask other data using shapefile

reference:
https://gis.stackexchange.com/questions/354782/masking-netcdf-time-series-data-from-shapefile-using-python
'''
import xarray as xr
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend
import concurrent.futures