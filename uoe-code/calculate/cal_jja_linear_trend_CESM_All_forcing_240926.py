'''
2024-9-26
This script is to calculate the 50 years trend in Indian in CESM control experiment
'''
import xarray as xr
import numpy as np
import sys
import os
from scipy.stats import t
from scipy.stats import bootstrap
import matplotlib.pyplot as plt

sys.path.append('/home/sun/uoe-code/module/')
from module_sun import cal_xydistance

import concurrent.futures

# ============================ File Location ==================================

file_CESM = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc")

file_GPCC = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/JJAS_GPCC_mean_update.nc")

key_area = [20, 28, 76, 87]

file_select = file_CESM.sel(time=slice(1901, 1955), lon=slice(76, 87), lat=slice(20, 28))

avg = np.average(np.average(file_select['PRECT_JJA_BTAL'].data, axis=1), axis=1)

slope, intercept = np.polyfit(np.linspace(1901, 1955, 55), avg, 1)

# 输出斜率
print(f"斜率: {slope}")