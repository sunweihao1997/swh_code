'''
2025-5-6
This script is used to calculate the pre-processed data for the streamfunction calculation

relevant link:
https://www.notion.so/ERL-Fig2a-850-hPa-replace-SLP-with-streamfunction-1ebd5b19b11d80dca81ae12f2d26c7ea?pvs=4
'''
import xarray as xr
import numpy as np
import os
import sys

data1 = xr.open_dataset('/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTAL_JJA_U_ensemble.nc').sel(lev=850)
data2 = xr.open_dataset('/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTAL_JJA_V_ensemble.nc').sel(lev=850)
data3 = xr.open_dataset('/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTALnEU_JJA_U_ensemble.nc').sel(lev=850)
data4 = xr.open_dataset('/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTALnEU_JJA_V_ensemble.nc').sel(lev=850)

u_btal_average = (data1['JJA_U_1'].data + data1['JJA_U_2'].data + data1['JJA_U_3'].data + data1['JJA_U_4'].data + data1['JJA_U_5'].data + data1['JJA_U_6'].data + data1['JJA_U_7'].data + data1['JJA_U_8'].data) /8
v_btal_average = (data2['JJA_V_1'].data + data2['JJA_V_2'].data + data2['JJA_V_3'].data + data2['JJA_V_4'].data + data2['JJA_V_5'].data + data2['JJA_V_6'].data + data2['JJA_V_7'].data + data2['JJA_V_8'].data) /8
u_btalneu_average = (data3['JJA_U_1'].data + data3['JJA_U_2'].data + data3['JJA_U_3'].data + data3['JJA_U_4'].data + data3['JJA_U_5'].data + data3['JJA_U_6'].data + data3['JJA_U_7'].data + data3['JJA_U_8'].data) /8  
v_btalneu_average = (data4['JJA_V_1'].data + data4['JJA_V_2'].data + data4['JJA_V_3'].data + data4['JJA_V_4'].data + data4['JJA_V_5'].data + data4['JJA_V_6'].data + data4['JJA_V_7'].data + data4['JJA_V_8'].data) /8

u_btal_average[np.isnan(u_btal_average)] = 0
v_btal_average[np.isnan(v_btal_average)] = 0
u_btalneu_average[np.isnan(u_btalneu_average)] = 0
v_btalneu_average[np.isnan(v_btalneu_average)] = 0

data1['JJA_U_average_BTAL'] = xr.DataArray(data=u_btal_average, dims=["time", "lat", "lon"],
                                    coords=dict(
                                        lon=(["lon"], data1.lon.data),
                                        lat=(["lat"], data1.lat.data),
                                        time=(["time"], data1.time.data),
                                    ),
                                    attrs=dict(
                                        description="model ensemble average u",
                                    ),
                                    )
data1['JJA_V_average_BTAL'] = xr.DataArray(data=v_btal_average, dims=["time", "lat", "lon"],
                                    coords=dict(
                                        lon=(["lon"], data1.lon.data),
                                        lat=(["lat"], data1.lat.data),
                                        time=(["time"], data1.time.data),
                                    ),
                                    attrs=dict(
                                        description="model ensemble average u",
                                    ),
                                    )
data1['JJA_U_average_BTALnEU'] = xr.DataArray(data=u_btalneu_average, dims=["time", "lat", "lon"],
                                    coords=dict(
                                        lon=(["lon"], data1.lon.data),
                                        lat=(["lat"], data1.lat.data),
                                        time=(["time"], data1.time.data),
                                    ),
                                    attrs=dict(
                                        description="model ensemble average u",
                                    ),
                                    )
data1['JJA_V_average_BTALnEU'] = xr.DataArray(data=v_btalneu_average, dims=["time", "lat", "lon"],
                                    coords=dict(
                                        lon=(["lon"], data1.lon.data),
                                        lat=(["lat"], data1.lat.data),
                                        time=(["time"], data1.time.data),
                                    ),
                                    attrs=dict(
                                        description="model ensemble average u",
                                    ),
                                    )

#print(data1)
data1.to_netcdf('/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTAL_BTALnEU_JJA_UV_ensemble_average_850hpa.nc', mode='w')

print("YES")
#sys.exit("Succeed")