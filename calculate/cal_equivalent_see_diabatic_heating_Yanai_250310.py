'''
20250310
This script is to calculate the see heating for the ERA5 data
'''

import numpy as np
import xarray as xr
import os
import math

#data_path = '/data1/other_data/DataUpdate/ERA5/new-era5/monthly/multiple/'

file_test = xr.open_dataset("/home/sun/mydown/ERA5/monthly_pressure/ERA5_2002_monthly_pressure_UTVWZSH.nc")
plev      = file_test['level'].data #; print(plev)

expand_plev = np.broadcast_to(plev[np.newaxis, :, np.newaxis, np.newaxis],  (len(file_test['time'].data), len(plev), len(file_test['latitude'].data), len(file_test['longitude'].data)))

#print(expand_plev[5, 5, 50, :])
term1 = pow(expand_plev/1000, 0.286) #(p/p0)^k

def cal_heating(t, u, v, p, w):
    # input variable is 4D (time, level, latitude, longitude)
    # parameters
    # unit of P: hPa
    r = 287 ; cp = 1004 ; kappa = r/cp ; p0 = 1000

    # calculate the potential temperature
    #theta = t * math.pow(p0/expand_plev, kappa)
    theta = t * ((p0/expand_plev) ** kappa)

    # calculate dtheta to dt
    dtheta_dt = np.gradient(theta, axis=0, edge_order=1) / (30 * 24 * 3600)

    # calculate the advection term
    radium = 6371000 

    dlat   = np.gradient(file_test['latitude'].data) * (np.pi / 180) * radium  

    dlon   = np.gradient(file_test['longitude'].data) * (np.pi / 180) * radium * np.cos(np.radians(file_test['latitude'].data[:, np.newaxis]))

    grad_lat = np.gradient(theta, axis=2) / dlat[np.newaxis, np.newaxis, :, np.newaxis]

    grad_lon = np.gradient(theta, axis=3) / dlon[np.newaxis, np.newaxis, :, :]

    advection_term = (u * grad_lon + v * grad_lat)

    # vertical term
    grad_p   = np.gradient(theta, axis=1) / np.gradient(plev)[:, np.newaxis, np.newaxis]

    vertical_term = w / 100 * grad_p

    return cp * pow((expand_plev/p0), kappa) * (dtheta_dt + advection_term + vertical_term)

# ========== Start calculating =================
path0 = "/home/sun/mydown/ERA5/monthly_pressure/"
path_out = "/home/sun/mydown/ERA5/monthly_pressure_diabatic_heating/"

file_list = os.listdir(path0) ; file_list.sort()
file_list = [element for element in file_list if '.nc' in element]
print(len(file_list))
for ffff in file_list[40:]:
    f1 = xr.open_dataset(path0 + ffff) ; print(f"Now it is dealing with {ffff}")

    diabatic_heating = cal_heating(f1["t"].data, f1["u"].data, f1["v"].data, plev, f1["w"].data)

    ncfile  =  xr.Dataset(
        {
            "diabatic_heating": (["time", "level", "latitude", "longitude"], diabatic_heating),
        },
        coords={
            "time": (["time"], file_test['time'].data),
            "level": (["level"], file_test['level'].data),
            "latitude": (["latitude"], file_test['latitude'].data),
            "longitude": (["longitude"], file_test['longitude'].data),
        },
        )
    ncfile.attrs['description']  =  'Calculated by /home/sun/swh_code/calculate/cal_equivalent_see_diabatic_heating_Yanai_250310.py. This file is the monthly diabatic heating calculated by Yanai (1973)'
    ncfile.to_netcdf(path_out + ffff.replace("UTVWZSH", "diabatic_heating"))

