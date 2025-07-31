'''
2025-7-24
This script is to calculate the reading test index for FY4.
'''
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from netCDF4 import Dataset

file_name = "/home/sun/wd_14/data/data/download_data/Fengyun4/test/FY4B-_GIIRS-_N_REGC_1050E_L2-_AVP-_MULT_NUL_20250718235437_20250718235945_012KM_024V1.NC"

file0     = xr.open_dataset(file_name,)

#print(file0.AT_Prof.shape)
print(file0.AT_Prof.data[:, 20])
