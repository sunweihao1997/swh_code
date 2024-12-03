'''
2024-7-17
This script is to calculate the diagnose quantity of the air-sea interaction
'''
import xarray as xr
import numpy as np
import os
import sys

# ============= File Information ===============
file_path = "/home/sun/data/merra2_2m_selection/"
list_all  = os.listdir(file_path)  ; list_all.sort()


# ============ Calculation 1. daymean ============
#end_path0 = "/home/sun/mydown/MERRA2_2m_daymean/"
#for ff in list_all:
#    if ff[-2:] == 'nc':
#        os.system("cdo daymean " + file_path + ff + " " + end_path0 + ff)
#
#        print(f"Successfully deal with {ff}")
def calculate_saturation_vapor_pressure(T):
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))

# 计算饱和比湿
def calculate_saturated_specific_humidity(T, p):
    e_s = calculate_saturation_vapor_pressure(T)
    q_s = 0.622 * e_s / (p - e_s)
    return q_s

# =========== Calculation 2. climatological ============
def cal_climate_value(file_path, file_list, start_year, end_year, varname):
    
    ref_file     = xr.open_dataset(file_path + file_list[5])
    lat          = ref_file.lat.data ; lon       = ref_file.lon.data

    # 1. Claim the average array
    climate_array= np.zeros((365, len(lat), len(lon)))

    for yyyy in range(start_year, end_year+1,):
        # 2. get the file list for single year
        list_single  = []
        for ff in file_list:
            if "Nx."+str(yyyy) in ff and "nc" in ff:
                list_single.append(ff)

        list_single.sort()
        list_single = list_single[:365]
        print(f"Now the climate file list {yyyy} has been completed, which contain {varname} {len(list_single)}")

        for i in range(len(list_single)):
            f1 = xr.open_dataset(file_path + list_single[i])

            climate_array[i] += (f1[varname].data[0] / (end_year - start_year + 1))

    return climate_array

def cal_climate_value_seahumidity(file_path, file_list, start_year, end_year, tvarname, pvarname):
    
    ref_file     = xr.open_dataset(file_path + file_list[5])
    lat          = ref_file.lat.data ; lon       = ref_file.lon.data

    # 1. Claim the average array
    climate_array= np.zeros((365, len(lat), len(lon)))

    for yyyy in range(start_year, end_year+1,):
        # 2. get the file list for single year
        list_single  = []
        for ff in file_list:
            if "Nx."+str(yyyy) in ff and "nc" in ff:
                list_single.append(ff)

        list_single.sort()
        list_single = list_single[:365]
        print(f"Now the climate file list {yyyy} has been completed, which contain {tvarname} {len(list_single)}")

        for i in range(len(list_single)):
            f1 = xr.open_dataset(file_path + list_single[i])

            climate_array[i] += (calculate_saturated_specific_humidity(f1[tvarname].data[0] - 273.15, f1[pvarname].data[0]/100) / (end_year - start_year + 1))

    return climate_array

# =========== Calculation 3. Anomaly ============
def cal_anomaly_value(file_path, file_list, start_year, end_year, varname, climate_value):
    
    ref_file     = xr.open_dataset(file_path + file_list[5])
    lat          = ref_file.lat.data ; lon       = ref_file.lon.data

    # 1. Claim the average array
    anomaly_array= np.zeros((end_year - start_year + 1, 365, len(lat), len(lon)))

    for yyyy in range(start_year, end_year+1,):
        # 2. get the file list for single year
        list_single  = []
        for ff in file_list:
            if "Nx."+str(yyyy) in ff and "nc" in ff:
                list_single.append(ff)

        list_single.sort()
        list_single = list_single[:365]
        print(f"Now the abnormal file list {yyyy} has been completed, which contain {varname} {len(list_single)}")

        for i in range(len(list_single)):
            f1 = xr.open_dataset(file_path + list_single[i])

            anomaly_array[yyyy - start_year, i] += (f1[varname].data[0] - climate_value[i])

    return anomaly_array

def cal_anomaly_value_seahumidity(file_path, file_list, start_year, end_year, tvarname, pvarname, climate_value):
    
    ref_file     = xr.open_dataset(file_path + file_list[5])
    lat          = ref_file.lat.data ; lon       = ref_file.lon.data

    # 1. Claim the average array
    anomaly_array= np.zeros((end_year - start_year + 1, 365, len(lat), len(lon)))

    for yyyy in range(start_year, end_year+1,):
        # 2. get the file list for single year
        list_single  = []
        for ff in file_list:
            if "Nx."+str(yyyy) in ff and "nc" in ff:
                list_single.append(ff)

        list_single.sort()
        list_single = list_single[:365]
        print(f"Now the abnormal file list {yyyy} has been completed, which contain {tvarname} {len(list_single)}")

        for i in range(len(list_single)):
            f1 = xr.open_dataset(file_path + list_single[i])

            anomaly_array[yyyy - start_year, i] += (calculate_saturated_specific_humidity(f1[tvarname].data[0] - 273.15, f1[pvarname].data[0]/100) - climate_value[i])

    return anomaly_array   



def main():
    # 1. Get the climate value
#    t2m  =  cal_climate_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "T2M")
#    ts   =  cal_climate_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "TS")
#    u2m  =  cal_climate_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "U2M")
#    v2m  =  cal_climate_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "V2M")
#    q2m  =  cal_climate_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "QV2M")
#    slp  =  cal_climate_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "SLP")
    qs   =  cal_climate_value_seahumidity('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "TS", "SLP")

    # 2. get the anomaly value
#    t2m_anomaly = cal_anomaly_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "T2M", t2m)
#    ts_anomaly  = cal_anomaly_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "TS",  ts)
#    u2m_anomaly = cal_anomaly_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "U2M", u2m)
#    v2m_anomaly = cal_anomaly_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "V2M", v2m)
#    q2m_anomaly = cal_anomaly_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "QV2M", q2m)
#    slp_anomaly = cal_anomaly_value('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "SLP", slp)
    qs_anomaly  = cal_anomaly_value_seahumidity('/home/sun/data/merra2_2m_selection/', list_all, 1980, 2019, "TS", "SLP", qs)
#    sys.exit()

    ref_file    = xr.open_dataset("/home/sun/data/merra2_2m_selection/MERRA2_300.tavg1_2d_slv_Nx.20020312.SUB.nc")
    # 3. Write to ncfile
    ncfile  =  xr.Dataset(
                {
#                    "t2m": (["day", "lat", "lon"], t2m),
#                    "ts":  (["day", "lat", "lon"], ts),
#                    "u2m": (["day", "lat", "lon"], u2m),
#                    "v2m": (["day", "lat", "lon"], v2m),
#                    "q2m": (["day", "lat", "lon"], q2m),
#                    "slp": (["day", "lat", "lon"], slp),
                    "qs":  (["day", "lat", "lon"], qs),
                    "qs_anomaly":  (["year", "day", "lat", "lon"], qs_anomaly),
#                    "slp_anomaly": (["year", "day", "lat", "lon"], slp_anomaly),
#                    "q2m_anomaly": (["year", "day", "lat", "lon"], q2m_anomaly),
#                    "t2m_anomaly": (["year", "day", "lat", "lon"], t2m_anomaly),
#                    "ts_anomaly":  (["year", "day", "lat", "lon"], ts_anomaly),
#                    "u2m_anomaly": (["year", "day", "lat", "lon"], u2m_anomaly),
#                    "v2m_anomaly": (["year", "day", "lat", "lon"], v2m_anomaly),
                },
                    coords={
                        "lat":  (["lat"],  ref_file.lat.data),
                        "lon":  (["lon"],  ref_file.lon.data),
                        "day":  (["day"],  np.linspace(1,365,365)),
                        "year": (["year"], np.linspace(1980,2019,2019-1980+1)),
                    },
                    )

    ncfile.attrs['description'] = 'created on 2024-7-17. This file contains climatology and abnormal value for some variables'

    ncfile.to_netcdf('/home/sun/data/climate_data/air_sea_interaction/climate_abnormal_qs.nc')



if __name__ == '__main__':
    main()