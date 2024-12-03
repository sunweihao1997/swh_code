'''
2023-11-6
This script is to extract only Indian region precipitation use precipitation and shapefile
'''
from matplotlib import projections
import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import cartopy

sys.path.append("/home/sun/mycode/module/")
from module_sun import *

sys.path.append("/home/sun/mycode/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import set_cartopy_tick,save_fig
from paint_lunwen_version3_0_fig2a_tem_gradient_20220426 import add_text

import geopandas
import rioxarray
from shapely.geometry import mapping

prect  =  xr.open_dataset('/home/sun/data/long_term_precipitation/Precipitation_single_NCEP_1x1_1891_2020_20231030.nc')
prect.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
prect.rio.write_crs("epsg:4326", inplace=True)

Indian_shape = geopandas.read_file('/home/sun/data/shapefile/IND_adm0.shp', crs="epsg:4326")

clipped = prect.rio.clip(Indian_shape.geometry.apply(mapping), Indian_shape.crs, drop=False)

clipped.to_netcdf("test.nc")