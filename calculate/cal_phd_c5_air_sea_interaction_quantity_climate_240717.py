'''
2024-7-17
This script is to calculate some variables in air-sea-interaction, including UdT and the other three conbinations
'''
import xarray as xr
import numpy as np
import os

# ======== File Information ==========
file0      = xr.open_dataset("/home/sun/data/climate_data/air_sea_interaction/climate_abnormal_u2m_v2m_t2m_ts.nc").sel(lat=slice(0, 5), lon=slice(55, 90))

onset_data = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/onsetdate.nc")

dates      = onset_data["bob_onset_date"].data
#print(dates)
# ======= End of file Information ======

# ====== calculation 1. average udT ========
udt        = np.sqrt(file0['u2m'].data **2 +  file0['v2m'].data**2) * (file0['t2m'].data - file0['ts'].data)

# ====== calculation 2. u'dT ==============
#u0dt       = np.sqrt(file0['u2m_anomaly'].data **2 +  file0['v2m_anomaly'].data**2) * (file0['t2m'].data - file0['ts'].data)

# ====== calculation 3. udT' ==============
udt0       = np.sqrt(file0['u2m'].data **2 + file0['v2m'].data**2) * (file0['t2m_anomaly'].data - file0['ts_anomaly'].data)

# ====== calculation 4. u'dT' =============
#u0dt0      = np.sqrt(file0['u2m_anomaly'].data **2 + file0['v2m_anomaly'].data**2) * (file0['t2m_anomaly'].data - file0['ts_anomaly'].data)
#print(u0dt0.shape)

#for i in range(40):
#    for j in range(365):
#        for k in range(6):
#            for z in range(36):
#                if file0['u2m_anomaly'].data[i, j, k, z] > 0 and file0['v2m_anomaly'].data[i, j, k, z] > 0:
#                    continue
#                elif file0['u2m_anomaly'].data[i, j, k, z] < 0 and file0['v2m_anomaly'].data[i, j, k, z] < 0:
#                    u0dt[i, j, k, z]  = -1 * u0dt[i, j, k, z]
#                    u0dt0[i, j, k, z] = -1 * u0dt0[i, j, k, z]
#                    continue
#                elif file0['u2m_anomaly'].data[i, j, k, z] < 0 and file0['v2m_anomaly'].data[i, j, k, z] > 0:
#                    if np.abs(file0['u2m_anomaly'].data[i, j, k, z]) > np.abs(file0['v2m_anomaly'].data[i, j, k, z]):
#                        u0dt[i, j, k, z]       = -1 * np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file0['t2m'].data[j, k, z] - file0['ts'].data[j, k, z])
#
#                        u0dt0[i, j, k, z]      = -1 * np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file0['t2m_anomaly'].data[i, j, k, z] - file0['t2m_anomaly'].data[i, j, k, z])
#                    else:
#                        u0dt[i, j, k, z]       = np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file0['t2m'].data[j, k, z] - file0['ts'].data[j, k, z])
#
#                        u0dt0[i, j, k, z]      = np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file0['t2m_anomaly'].data[i, j, k, z] - file0['t2m_anomaly'].data[i, j, k, z])
#                elif file0['u2m_anomaly'].data[i, j, k, z] > 0 and file0['v2m_anomaly'].data[i, j, k, z] < 0:
#                    if np.abs(file0['u2m_anomaly'].data[i, j, k, z]) > np.abs(file0['v2m_anomaly'].data[i, j, k, z]):
#                        u0dt[i, j, k, z]       = np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file0['t2m'].data[j, k, z] - file0['ts'].data[j, k, z])
#
#                        u0dt0[i, j, k, z]      = np.sqrt(file0['u2m_anomaly'].data[i, j, k, z] **2 -  file0['v2m_anomaly'].data[i, j, k, z]**2) * (file0['t2m_anomaly'].data[i, j, k, z] - file0['t2m_anomaly'].data[i, j, k, z])
#                    else:
#                        #print(file0['t2m'].data.shape)
#                        u0dt[i, j, k, z]       = -1 * np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file0['t2m'].data[j, k, z] - file0['ts'].data[j, k, z])
#
#                        u0dt0[i, j, k, z]      = -1 * np.sqrt(file0['v2m_anomaly'].data[i, j, k, z]**2 - file0['u2m_anomaly'].data[i, j, k, z] **2) * (file0['t2m_anomaly'].data[i, j, k, z] - file0['t2m_anomaly'].data[i, j, k, z])
 
#print("Done")
## Write to ncfile
#ncfile  =  xr.Dataset(
#{
#    "u0dt":  (["year", "day", "lat", "lon"],  u0dt),
#    "u0dt0": (["year", "day", "lat", "lon"], u0dt0),
#},
#coords={
#    "year": (["year"], file0.year.data),
#    "day":  (["day"],  file0.day.data),
#    "lat":  (["lat"],  file0.lat.data),
#    "lon":  (["lon"],  file0.lon.data),
#},
#)
#ncfile.attrs['description']  =  'cal_phd_c5_air_sea_interaction_quantity_early_240717.py'
#ncfile.to_netcdf("/home/sun/data/process/air_sea_interaction_quantity.nc")

f2 = xr.open_dataset("/home/sun/data/process/air_sea_interaction_quantity.nc")
u0dt  = f2.u0dt.data
u0dt0 = f2.u0dt0.data

# ====== calculation 5. composite ==========
shf_climate = np.zeros((4, 61, len(file0.lat.data), len(file0.lon.data))) ; num_climate = 0
shf_early   = np.zeros((4, 61, len(file0.lat.data), len(file0.lon.data))) ; num_early   = 0
shf_late    = np.zeros((4, 61, len(file0.lat.data), len(file0.lon.data))) ; num_late    = 0

for i in range(40):
    day0 = dates[i]

    if day0 <= np.average(dates) - 1*np.std(dates):
        num_early += 1

        for j in range(-30, 31):
            shf_early[0, j + 30]   += udt[j + day0]
            shf_early[0+1, j + 30] += u0dt[i, j + day0]
            shf_early[1+1, j + 30] += udt0[i, j + day0]
            shf_early[2+1, j + 30] += u0dt0[i, j + day0]

    if day0 >= np.average(dates) + 1*np.std(dates):
        num_late  += 1

        for j in range(-30, 31):
            shf_late[0, j + 30]   += udt[j + day0]
            shf_late[0+1, j + 30] += u0dt[i, j + day0]
            shf_late[1+1, j + 30] += udt0[i, j + day0]
            shf_late[2+1, j + 30] += u0dt0[i, j + day0]

    num_climate += 1
    for j in range(-30, 31):
        shf_climate[0, j + 30]   += udt[j + day0]
        shf_climate[0+1, j + 30] += u0dt[i, j + day0]
        shf_climate[1+1, j + 30] += udt0[i, j + day0]
        shf_climate[2+1, j + 30] += u0dt0[i, j + day0]

shf_early /= num_early ; shf_late /= num_late ; shf_climate /= num_climate

start0 = 2 ; end0 =30


# ========== Plot the figure ==============
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(6, 12))

axs[0].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[0, 0:5], axis=0), axis=0),     color='k', marker='s',    lw=1.5, markersize=5, alpha=1, label = "udT")
axs[0].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[1, 0:5], axis=0), axis=0),     color='r', marker='^',    lw=1.5, markersize=5, alpha=1, label = "u'dT")
axs[0].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[2, 0:5], axis=0), axis=0),     color='b', marker='X',    lw=1.5, markersize=5, alpha=1, label = "udT'")
axs[0].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[3, 0:5], axis=0), axis=0),     color='grey', marker='X', lw=1.5, markersize=5, alpha=1, label = "u'dT'")
axs[0].set_xticks(np.linspace(55, 90, 8))
axs[0].set_xticklabels(['55E', '60E', '65E','70E','75E','80E','85E','90E',])
axs[0].set_title("Period1 [D0-30 to D0-20]", loc='right')

axs[0].set_ylim((-2.5, 5.5))

axs[1].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[0, 15:22], axis=0), axis=0),     color='k', marker='s',    lw=1.5, markersize=5, alpha=1, label = "udT")
axs[1].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[1, 15:22], axis=0), axis=0),     color='r', marker='^',    lw=1.5, markersize=5, alpha=1, label = "u'dT")
axs[1].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[2, 15:22], axis=0), axis=0),     color='b', marker='X',    lw=1.5, markersize=5, alpha=1, label = "udT'")
axs[1].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[3, 15:22], axis=0), axis=0),     color='grey', marker='X', lw=1.5, markersize=5, alpha=1, label = "u'dT'")
axs[1].set_ylim((-2.5, 5.5))
axs[1].set_xticks(np.linspace(55, 90, 8))
axs[1].set_xticklabels(['55E', '60E', '65E','70E','75E','80E','85E','90E',])
axs[1].set_title("Period2 [D0-20 to D0-10]", loc='right')

axs[2].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[0, 23:30], axis=0), axis=0),     color='k', marker='s',    lw=1.5, markersize=5, alpha=1, label = "udT")
axs[2].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[1, 23:30], axis=0), axis=0),     color='r', marker='^',    lw=1.5, markersize=5, alpha=1, label = "u'dT")
axs[2].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[2, 23:30], axis=0), axis=0),     color='b', marker='X',    lw=1.5, markersize=5, alpha=1, label = "udT'")
axs[2].plot(np.linspace(55, 90, 36), -1 * np.average(np.average(shf_climate[3, 23:30], axis=0), axis=0),     color='grey', marker='X', lw=1.5, markersize=5, alpha=1, label = "u'dT'")
axs[2].set_ylim((-2.5, 5.5))
axs[2].set_xticks(np.linspace(55, 90, 8))
axs[2].set_xticklabels(['55E', '60E', '65E','70E','75E','80E','85E','90E',])
axs[2].set_title("Period3 [D0-10 to D0]", loc='right')

axs[0].legend(loc='upper left')

plt.savefig("/home/sun/paint/phd/climate_shf_collapse_p123.pdf")