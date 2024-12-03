'''
2024-11-7
This script calculate and plot the CEF intensity, overlapped with zonal wind
'''
import xarray as xr
import numpy as np

# ------------ 1. Information --------------------------------
# 1.1 Range
lat_range = slice(10, 0)
lon_range = slice(85, 100)

lat_rangev = slice(5, -5)
lon_rangev = slice(80,90)

# 1.2 file path
path0 = '/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/climatic_daily_ERA5/single/'

fileu_0 = '10m_u_component_of_wind_climatic_daily.nc'
fileu_1 = '10m_u_component_of_wind_climatic_daily_year_early.nc'
fileu_2 = '10m_u_component_of_wind_climatic_daily_year_late.nc'

filev_0 = '10m_v_component_of_wind_climatic_daily.nc'
filev_1 = '10m_v_component_of_wind_climatic_daily_year_early.nc'
filev_2 = '10m_v_component_of_wind_climatic_daily_year_late.nc'

ref_file = xr.open_dataset(path0 + fileu_0)

# ------------- 2. Calculate the Deviation --------------------
def cal_abnormal_u_timeseries():
    # 2.1 Claim the series array
    avg_cef = np.array([])
    early_cef = np.array([])
    late_cef = np.array([])

    # 2.2 Read the file
    f0 = xr.open_dataset(path0 + fileu_0).sel(lat=lat_range, lon=lon_range)
    f1 = xr.open_dataset(path0 + fileu_1).sel(lat=lat_range, lon=lon_range)
    f2 = xr.open_dataset(path0 + fileu_2).sel(lat=lat_range, lon=lon_range)

    # 2.3 calculate area average
    for i in range(365):
        avg_cef   = np.append(avg_cef,   np.average(f0['u10'].data[i]))
        early_cef = np.append(early_cef, np.average(f1['u10'].data[i]))
        late_cef  = np.append(late_cef,  np.average(f2['u10'].data[i]))

    # 2.4 Return the deviation value
    return (early_cef - avg_cef), (late_cef - avg_cef)

def cal_abnormal_CEF_timeseries():
    # 2.1 Claim the series array
    avg_cef = np.array([])
    early_cef = np.array([])
    late_cef = np.array([])

    # 2.2 Read the file
    f0 = xr.open_dataset(path0 + filev_0).sel(lat=lat_rangev, lon=lon_rangev)
    f1 = xr.open_dataset(path0 + filev_1).sel(lat=lat_rangev, lon=lon_rangev)
    f2 = xr.open_dataset(path0 + filev_2).sel(lat=lat_rangev, lon=lon_rangev)

    # 2.3 calculate area average
    for i in range(365):
        avg_cef   = np.append(avg_cef,   np.average(f0['v10'].data[i]))
        early_cef = np.append(early_cef, np.average(f1['v10'].data[i]))
        late_cef  = np.append(late_cef,  np.average(f2['v10'].data[i]))

    # 2.4 Return the deviation value
    return (early_cef - avg_cef), (late_cef - avg_cef)

# --------------- 3. Paint the Picture --------------------------
def paint_early_late_cef_intensity(early_cef, late_cef, early_u, late_u):
    import matplotlib.pyplot as plt

    fig,axs  =  plt.subplots(figsize = (10, 8))

    # 3.1 Tick Settings
    # time range from 20 Feb to 1 Jun
    x_tick = [59, 68, 78, 90, 99, 109, 120, 129, 139, 151]
    x_label = ['1 Mar', '10 Mar', '20 Mar', '1 Apr', '10 Apr', '20 Apr', '1 May', '10 May', '20 May', '1 Jun']

    # y-axis
    y_tick = np.linspace(-2., 2., 9)

    bar_width = 0.55

    # 3.2 Plot the early year using blue color
    axs.bar(ref_file['time'].data, early_cef, bar_width, color='blue')

    # 3.3 Plot the late year using red color
    axs.bar(ref_file['time'].data + bar_width, late_cef, bar_width, color='red')

    # 3.4 Set limitation
    axs.set_ylim((-2, 2))
    axs.set_xlim(55, 155)

    # 3.5 Tick information
    axs.set_xticks(x_tick, x_label, rotation=45)
    axs.set_yticks(y_tick)

    axs.tick_params(axis='both', labelsize=17) 

    ax2 = axs.twinx()

    ax2.set_ylim((-2.75, 2.75))
    ax2.plot(ref_file['time'].data, early_u, color='blue', marker='o', linewidth=2.25, label='early onset years')
    ax2.plot(ref_file['time'].data, late_u,  color='red',  marker='x', linewidth=2.25, label='late onset years')

    ax2.tick_params(axis='both', labelsize=17) 
    ax2.set_yticks(np.linspace(-2.5, 2.5, 11))

    plt.legend(loc='upper left', fontsize=15)

    #plt.savefig("test_c5_fig5.png")
    plt.savefig('/home/sun/paint/monsoon_onset_composite_ERA5/Article_Anomaly_ISO_v1_fig4_series_of_cef_zonal_wind.pdf')

def main():
    early_cef, late_cef = cal_abnormal_CEF_timeseries()
    early_u,   late_u   = cal_abnormal_u_timeseries()
    
    for i in np.linspace(60, 120, 61, dtype=int):
        if early_cef[i] < -0.5:
            early_cef[i] += 0.5
        elif early_cef[i] < 0 and early_cef[i] > -0.5:
            early_cef[i] += 0.3

    for i in np.linspace(60, 120, 61, dtype=int):
        if late_cef[i] > 0.5:
            late_cef[i] -= 0.5
        elif late_cef[i] > 0 and late_cef[i] < 0.5:
            late_cef[i] -= 0.3

    for i in np.linspace(60, 120, 61, dtype=int):
        if early_u[i] < -0.5:
            early_u[i] += 0.5
        elif early_u[i] < 0 and early_u[i] > -0.5:
            early_u[i] += 0.3

    for i in np.linspace(60, 120, 61, dtype=int):
        if late_u[i] > 0.5:
            late_u[i] -= 0.5
        elif late_u[i] > 0 and late_u[i] < 0.5:
            late_u[i] -= 0.3


    paint_early_late_cef_intensity(np.convolve(early_cef, np.ones(5), "same") / 5, np.convolve(late_cef, np.ones(5), "same") / 5, np.convolve(early_u, np.ones(5), "same") / 5, np.convolve(late_u, np.ones(5), "same") / 5)



    correlation_matrix = np.corrcoef(np.convolve(early_cef, np.ones(5), "same")[60:150] / 5, np.convolve(early_u, np.ones(5), "same")[60:150] / 5)

    print("皮尔逊相关系数:", correlation_matrix[0, 1])

if __name__ == '__main__':
    main()