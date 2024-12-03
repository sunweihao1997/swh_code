'''
2024-7-17
This script is to calculate some variables in air-sea-interaction, including UdT and the other three conbinations
'''
import xarray as xr
import numpy as np
import os
import sys

# ======== File Information ==========
file0      = xr.open_dataset("/home/sun/data/climate_data/air_sea_interaction/climate_abnormal_u2m_v2m_t2m_ts_q2m.nc").sel(lat=slice(0, 5), lon=slice(55, 90))
file1      = xr.open_dataset("/home/sun/data/climate_data/air_sea_interaction/climate_abnormal_qs.nc").sel(lat=slice(0, 5), lon=slice(55, 90))

onset_data = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/onsetdate.nc")

dates      = onset_data["bob_onset_date"].data
#print(dates)
# ======= End of file Information ======
#print(file0['slp'].data)

def calculate_saturation_vapor_pressure(T):
    return 6.112 * np.exp((17.67 * T) / (T + 243.5))

# 计算饱和比湿
def calculate_saturated_specific_humidity(T, p):
    e_s = calculate_saturation_vapor_pressure(T)
    q_s = 0.622 * e_s / (p - e_s)
    return q_s


# ====== calculation 1. average udT ========
#print(np.nanmax(file0['q2m'].data)) # unit kg/kg
#sys.exit()
udq        = np.sqrt(file0['u2m'].data **2 +  file0['v2m'].data**2) * (file1['qs'].data - file0['q2m'].data)
#print(file1['qs'].data)
#print(file0['q2m'].data)
# ====== calculation 2. u'dT ==============
u0dq       = np.sqrt(file0['u2m_anomaly'].data **2 +  file0['v2m_anomaly'].data**2) * (file1['qs'].data - file0['q2m'].data)

# ====== calculation 3. udT' ==============
udq0       = np.sqrt(file0['u2m'].data **2 + file0['v2m'].data**2) * (file1['qs_anomaly'].data - file0['q2m_anomaly'].data)

# ====== calculation 4. u'dT' =============
u0dq0      = np.sqrt(file0['u2m_anomaly'].data **2 + file0['v2m_anomaly'].data**2) * (file1['qs_anomaly'].data - file0['q2m_anomaly'].data)
#print(u0dq0.shape)

#print(calculate_saturated_specific_humidity(file0['ts'].data-273.15, file0['slp'].data/100) - file0['q2m_anomaly'].data)
#sys.exit()
#for i in range(40):
#    print(f'Now it is the year {i}')
#    for j in range(365):
#        for k in range(6):
#            for z in range(36):
#                if file0['u2m_anomaly'].data[i, j, k, z] > 0 and file0['v2m_anomaly'].data[i, j, k, z] > 0:
#                    continue
#                elif file0['u2m_anomaly'].data[i, j, k, z] < 0 and file0['v2m_anomaly'].data[i, j, k, z] < 0:
#                    u0dq[i, j, k, z]  = -1 * u0dq[i, j, k, z]
#                    u0dq0[i, j, k, z] = -1 * u0dq0[i, j, k, z]
#                    continue
#                elif file0['u2m_anomaly'].data[i, j, k, z] < 0 and file0['v2m_anomaly'].data[i, j, k, z] > 0:
#                    if np.abs(file0['u2m_anomaly'].data[i, j, k, z]) > np.abs(file0['v2m_anomaly'].data[i, j, k, z]):
#                        u0dq[i, j, k, z]       = -1 * np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file1['qs'].data[j, k, z] - file0['q2m'].data[j, k, z])
#
#                        u0dq0[i, j, k, z]      = -1 * np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file1['qs_anomaly'].data[i, j, k, z] - file0['q2m_anomaly'].data[i, j, k, z])
#                    else:
#                        u0dq[i, j, k, z]       = np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file1['qs'].data[j, k, z] - file0['q2m'].data[j, k, z])
#
#                        u0dq0[i, j, k, z]      = np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file1['qs_anomaly'].data[i, j, k, z] - file0['q2m_anomaly'].data[i, j, k, z])
#                elif file0['u2m_anomaly'].data[i, j, k, z] > 0 and file0['v2m_anomaly'].data[i, j, k, z] < 0:
#                    if np.abs(file0['u2m_anomaly'].data[i, j, k, z]) > np.abs(file0['v2m_anomaly'].data[i, j, k, z]):
#                        u0dq[i, j, k, z]       = np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file1['qs'].data[j, k, z] - file0['q2m'].data[j, k, z])
#
#                        u0dq0[i, j, k, z]      = np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file1['qs_anomaly'].data[i, j, k, z] - file0['q2m_anomaly'].data[i, j, k, z])
#                    else:
#                        #print(file0['q2m'].data.shape)
#                        u0dq[i, j, k, z]       = -1 * np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file1['qs'].data[j, k, z] - file0['q2m'].data[j, k, z])
#
#                        u0dq0[i, j, k, z]      = -1 * np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file1['qs_anomaly'].data[i, j, k, z] - file0['q2m_anomaly'].data[i, j, k, z])
#
#print("Done")
## Write to ncfile
#ncfile  =  xr.Dataset(
#{
#    "u0dq":  (["year", "day", "lat", "lon"],  u0dq),
#    "u0dq0": (["year", "day", "lat", "lon"], u0dq0),
#},
#coords={
#    "year": (["year"], file0.year.data),
#    "day":  (["day"],  file0.day.data),
#    "lat":  (["lat"],  file0.lat.data),
#    "lon":  (["lon"],  file0.lon.data),
#},
#)
#ncfile.attrs['description']  =  'cal_phd_c5_air_sea_interaction_quantity_early_240717.py'
#ncfile.to_netcdf("/home/sun/data/process/air_sea_interaction_quantity_slhf.nc")

f2 = xr.open_dataset("/home/sun/data/process/air_sea_interaction_quantity_slhf.nc")
u0dq = f2.u0dq.data
u0dq0= f2.u0dq0.data
#print(np.average(u0dq.shape)
#print(np.average(udq))

# ====== calculation 5. composite ==========
lhf_climate = np.zeros((4, 61, len(file0.lat.data), len(file0.lon.data))) ; num_climate = 0
lhf_early   = np.zeros((4, 61, len(file0.lat.data), len(file0.lon.data))) ; num_early   = 0
lhf_late    = np.zeros((4, 61, len(file0.lat.data), len(file0.lon.data))) ; num_late    = 0

for i in range(40):
    day0 = dates[i]

    if day0 <= np.average(dates) - 1*np.std(dates):
        num_early += 1

        for j in range(-30, 31):
            lhf_early[0, j + 30]   += udq[j + day0]
            lhf_early[0+1, j + 30] += u0dq[i, j + day0]
            lhf_early[1+1, j + 30] += udq0[i, j + day0]
            lhf_early[2+1, j + 30] += u0dq0[i, j + day0]

    if day0 >= np.average(dates) + 1*np.std(dates):
        num_late  += 1

        for j in range(-30, 31):
            lhf_late[0, j + 30]   += udq[j + day0]
            lhf_late[0+1, j + 30] += u0dq[i, j + day0]
            lhf_late[1+1, j + 30] += udq0[i, j + day0]
            lhf_late[2+1, j + 30] += u0dq0[i, j + day0]

    num_climate += 1
    for j in range(-30, 31):
        lhf_climate[0, j + 30]   += udq[j + day0]
        lhf_climate[0+1, j + 30] += u0dq[i, j + day0]
        lhf_climate[1+1, j + 30] += udq0[i, j + day0]
        lhf_climate[2+1, j + 30] += u0dq0[i, j + day0]

lhf_early /= num_early ; lhf_late /= num_late ; lhf_climate /= num_climate

start0 = 2 ; end0 =30

print(np.average(np.average(lhf_climate[1, 0:10], axis=0), axis=0))
# test
#print(np.average(np.average(lhf_early[1, 10:18], axis=0), axis=0))
#print(np.average(np.average(lhf_early[3, 10:18], axis=0), axis=0))
#sys.exit()

# ========== Plot the figure ==============
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(6, 12))

axs[0].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[0, 0:8], axis=0), axis=0),     color='k', marker='s',    lw=1.5, markersize=5, alpha=1, label = "udq")
axs[0].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[1, 0:8], axis=0), axis=0),     color='r', marker='^',    lw=1.5, markersize=5, alpha=1, label = "u'dq")
axs[0].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[2, 0:8], axis=0), axis=0),     color='b', marker='X',    lw=1.5, markersize=5, alpha=1, label = "udq'")
axs[0].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[3, 0:8], axis=0), axis=0),     color='grey', marker='X', lw=1.5, markersize=5, alpha=1, label = "u'dq'")
axs[0].set_xticks(np.linspace(55, 90, 8))
axs[0].set_xticklabels(['55E', '60E', '65E','70E','75E','80E','85E','90E',])
axs[0].set_title("Period1 [D0-30 to D0-20]", loc='right')

axs[0].set_ylim((-0.02, 0.035))

axs[1].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[0, 10:18], axis=0), axis=0),     color='k', marker='s',    lw=1.5, markersize=5, alpha=1, label = "udq")
axs[1].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[1, 10:18], axis=0), axis=0),     color='r', marker='^',    lw=1.5, markersize=5, alpha=1, label = "u'dq")
axs[1].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[2, 10:18], axis=0), axis=0),     color='b', marker='X',    lw=1.5, markersize=5, alpha=1, label = "udq'")
axs[1].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[3, 10:18], axis=0), axis=0),     color='grey', marker='X', lw=1.5, markersize=5, alpha=1, label = "u'dq'")
axs[1].set_ylim((-0.02, 0.035))
axs[1].set_xticks(np.linspace(55, 90, 8))
axs[1].set_xticklabels(['55E', '60E', '65E','70E','75E','80E','85E','90E',])
axs[1].set_title("Period2 [D0-20 to D0-10]", loc='right')

axs[2].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[0, 22:28], axis=0), axis=0),     color='k', marker='s',    lw=1.5, markersize=5, alpha=1, label = "udq")
axs[2].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[1, 22:28], axis=0), axis=0),     color='r', marker='^',    lw=1.5, markersize=5, alpha=1, label = "u'dq")
axs[2].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[2, 22:28], axis=0), axis=0),     color='b', marker='X',    lw=1.5, markersize=5, alpha=1, label = "udq'")
axs[2].plot(np.linspace(55, 90, 36), np.average(np.average(lhf_late[3, 22:28], axis=0), axis=0),     color='grey', marker='X', lw=1.5, markersize=5, alpha=1, label = "u'dq'")
axs[2].set_ylim((-0.02, 0.035))
axs[2].set_xticks(np.linspace(55, 90, 8))
axs[2].set_xticklabels(['55E', '60E', '65E','70E','75E','80E','85E','90E',])
axs[2].set_title("Period3 [D0-10 to D0]", loc='right')

axs[0].legend(loc='upper left')

plt.savefig("/home/sun/paint/phd/slhf_l_p123.pdf")