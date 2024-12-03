'''
2024-6-4
This script is to calculate the LSTC index using ERA5 data
'''
import numpy as np
import xarray as xr
import os 
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============= File Information ===================

data_path = "/home/sun/mydown/ERA5/monthly_single/"

# Define the range for calculating
lon_range_land = slice(70, 90)
lat_range_land = slice(20, 5)

lon_range_ocean= slice(70, 90)
lat_range_ocean= slice(20, -20)

# Mask file
mask_file = xr.open_dataset("/home/sun/data/mask/ERA5_land_sea_mask.nc")
mask      = mask_file.lsm.data[0]

# ==================================================

def calculate_land_sea_thermal_contrast(file0, land_lon, land_lat, ocean_lon, ocean_lat, varname):
    file_land = file0.sel(latitude=land_lat,  longitude=land_lon)
    file_ocean= file0.sel(latitude=ocean_lat, longitude=ocean_lon)

    mask_land = mask_file.sel(latitude=land_lat,  longitude=land_lon)
    mask_ocean= mask_file.sel(latitude=ocean_lat, longitude=ocean_lon)

    time      = file_land.time.data
    for tt in range(len(time)):
        file_land[varname].data[tt][mask_land['lsm'].data[0]<0.1]   = np.nan
        file_ocean[varname].data[tt][mask_ocean['lsm'].data[0]>0.1] = np.nan 

    lstc      = np.nanmean(np.nanmean(file_land[varname].data, axis=1), axis=1) - np.nanmean(np.nanmean(file_ocean[varname].data, axis=1), axis=1)

    return lstc

def calculate_maritime_continent_olr(file0,):
    file_maritime = file0.sel(latitude=slice(10, 0),  longitude=slice(110, 130))

    olr      = np.nanmean(np.nanmean(file_maritime['ttr'].data/86400, axis=1), axis=1)

    return olr

def calculate_maritime_continent_Africa_olr(file0,):
    file_maritime = file0.sel(latitude=slice(10, 0),  longitude=slice(110, 130))
    file_eastAf   = file0.sel(latitude=slice(10, 0),  longitude=slice(30, 45))

    olr      = np.nanmean(np.nanmean(file_maritime['ttr'].data/86400, axis=1), axis=1) - np.nanmean(np.nanmean(file_eastAf['ttr'].data/86400, axis=1), axis=1)

    return olr

if __name__ == "__main__":

    # =============== Use the T2M =================
    lstc_42year_7090 = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_7090 = np.append(lstc_42year_7090, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, lon_range_ocean, lat_range_ocean, 't2m'))

        #print(f'Finish the year {str(int(yy))}')

    lstc_42year_iob = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_iob = np.append(lstc_42year_iob, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, slice(40, 100), slice(20, -20), 't2m'))

        #print(f'Finish the year {str(int(yy))}')

    # =============== Use the skt =================
    lstc_42year_7090_ts = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_7090_ts = np.append(lstc_42year_7090_ts, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, lon_range_ocean, lat_range_ocean, 'skt'))

        #print(f'Finish the year {str(int(yy))}')

    lstc_42year_iob_ts = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_iob_ts = np.append(lstc_42year_iob_ts, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, slice(40, 100), slice(20, -20), 'skt'))

    # =============== Use the sp =================
    lstc_42year_7090_sp = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_7090_sp = np.append(lstc_42year_7090_sp, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, lon_range_ocean, lat_range_ocean, 'sp'))

        #print(f'Finish the year {str(int(yy))}')

    lstc_42year_iob_sp = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_iob_sp = np.append(lstc_42year_iob_sp, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, slice(40, 100), slice(20, -20), 'sp'))

    # =============== Use the psl =================
    lstc_42year_7090_psl = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_7090_psl = np.append(lstc_42year_7090_psl, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, lon_range_ocean, lat_range_ocean, 'msl'))

        #print(f'Finish the year {str(int(yy))}')

    lstc_42year_iob_psl = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        lstc_42year_iob_psl = np.append(lstc_42year_iob_psl, calculate_land_sea_thermal_contrast(f0, lon_range_land, lat_range_land, slice(40, 100), slice(20, -20), 'msl'))

    # =============== Use the Maritime OLR =================
    olr_42year_110130 = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        olr_42year_110130 = np.append(olr_42year_110130, calculate_maritime_continent_olr(f0,))

        #print(f'Finish the year {str(int(yy))}')

    # =============== Use the Difference between Maritime and Africa OLR ==============

    olr_42year_110130_3045 = np.array([])

    for yy in np.linspace(1980, 2021, 42):
        filename = "0000_single_month.nc".replace("0000", str(int(yy)))

        f0       = xr.open_dataset(data_path + filename)

        olr_42year_110130_3045 = np.append(olr_42year_110130_3045, calculate_maritime_continent_Africa_olr(f0,))


    #print(np.nanmax(lstc_42year_iob_ts))
    #print(np.nanmax(lstc_42year_iob))

    # Generate the timeaxis
    date_range = pd.date_range(start='1980-01-01', end='2021-12-31', freq='M')
    #print(date_range.shape)

    # Write to ncfile
    ncfile  =  xr.Dataset(
        {
            "LSTC_t2m_7090":  (["time"], lstc_42year_7090),
            "LSTC_t2m_IOB":   (["time"], lstc_42year_iob),
            "LSTC_ts_7090":   (["time"], lstc_42year_7090_ts),
            "LSTC_ts_IOB":   (["time"], lstc_42year_iob_ts),
            "LSTC_sp_7090":  (["time"], lstc_42year_7090_sp),
            "LSTC_sp_IOB":   (["time"], lstc_42year_iob_sp),
            "LSTC_psl_7090":   (["time"], lstc_42year_7090_psl),
            "LSTC_psl_IOB":   (["time"], lstc_42year_iob_psl),
            "OLR_maritime":   (["time"], olr_42year_110130),
            "OLR_mari_Afri":   (["time"], olr_42year_110130_3045),
        },
        coords=dict(
        time=("time", date_range),
    ),
        )

    ncfile.attrs['description']  =  'This file saves the LSTC calculating from the ERA5 data, for 7090 the ocean selection is 20to-20, while the iob means the area for IOB index which is (-20-20N, 40-100E)'
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc")