'''
2024-6-2
This script is to calculate the evolution of bandpass filtered precipitation in onset early/late years
'''
import xarray as xr
import numpy as np
from scipy.signal import butter, filtfilt
import sys

# ========== File Information =============

# data about onset early/late years
onset_day_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_include_early_late_years_new.nc")
print(onset_day_file.year_late.data)
onset_day_file_42 = onset_day_file.sel(year=slice(1980, 2021)) #42 years

# data about OLR
olr_bandpass_file = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_tp_bandpass_filter_20_70_new.nc").sel(lat=slice(15, 5), lon=slice(85, 100))    # best: lon=slice(80, 90)
olr_bandpass_file2= xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_tp_bandpass_filter_20_70_new.nc").sel(lat=slice(15, 5), lon=slice(85, 100))    # best: lon=slice(80, 87)
#print(np.average(np.average(olr_bandpass_file['tp_filter'].data, axis=1), axis=1)[0:365])
#sys.exit()
#print(olr_bandpass_file)

# ========================================
def cal_time_evolution_OLR(earlyfile, latefile, high_frq_varname, low_frq_varname, origin):
    # Verify how many years in each file
    early_number = int(earlyfile.time.data.shape[0]/365)
    late_number  = int(latefile.time.data.shape[0]/365)

    # Claim the array for saving the result
    OLR_early_high    = np.zeros((early_number, 365))
    OLR_late_high     = np.zeros((late_number,  365))

    OLR_early_low     = np.zeros((early_number, 365))
    OLR_late_low      = np.zeros((late_number,  365))

    OLR_origin            = np.zeros((42, 365))

    # Deal with the early year
    for i in range(early_number):
        OLR_early_high[i] = earlyfile[high_frq_varname].data[365*i:365*i+365]
        OLR_early_low[i]  = earlyfile[low_frq_varname].data[365*i:365*i+365]

    for i in range(late_number):
        OLR_late_high[i] = latefile[high_frq_varname].data[365*i:365*i+365]
        OLR_late_low[i]  = latefile[low_frq_varname].data[365*i:365*i+365]

    # Deal with the origin
    for i in range(42):
        #print(origin[high_frq_varname].data)
        OLR_origin[i]    = origin[high_frq_varname].data[365*i:365*i+365] + origin[low_frq_varname].data[365*i:365*i+365]

    return OLR_early_high, OLR_early_low, OLR_late_high, OLR_late_low, OLR_origin

def cal_time_evolution_OLR_2edition(file0, varname):
    ''' calculate average first and then filter '''
    # Verify how many years in each file
    year_number = int(file0.time.data.shape[0]/365)

    # Claim the array to save the value
    total_array   = np.zeros((year_number, 365))

    # Deal with the origin
    for i in range(year_number):
        #print(origin[high_frq_varname].data)
        total_array[i]    = np.average(np.average(file0[varname].data[365*i:365*i+365], axis=1), axis=1)

    avg_array             = np.average(total_array, axis=0)

    return avg_array, np.std(total_array, axis=0)

def screen_early_late(array_onset):
    early_years = np.array([])
    late_years  = np.array([])

    early_day = 0
    late_day  = 0

    std         = np.std(array_onset)

    for i in range(len(array_onset)):
        if array_onset[i] < np.average(array_onset) - std:
            early_years = np.append(early_years, i + 1980)

        elif array_onset[i] > np.average(array_onset) + std:
            late_years  = np.append(late_years, i + 1980)
        else:
            continue

    #print(np.average(array_onset))
    #print(late_years)
    return early_years, late_years

def plot_bandpass_olr(filter_avg, filter_std, time, day_notation):
    ''' This function is to plot the evolution of low and high frq OLR during early and late years'''
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    fig, axs = plt.subplots(1, 1, figsize=(13.5, 5))

    #plt.xticks(rotation=45)

    # ======== high-frq ==========
    ax0      = axs

    ax0.xaxis.set_major_locator(MultipleLocator(10))
    
    ax0.set_ylim((-5.5, 5.5))

    #ax0.set_title("8-20 band-pass", loc='left', fontsize=20)
    #ax0.plot(time, np.average(high_frq[0][:, 59:(59+92)], axis=0), color='blue')
    # ============ plot the avg about filter data ==============
    #ax0.plot(time, filter_avg[0][59:(59+92)] - filter_avg[2][59:(59+92)], linewidth=2.5, color='blue', label='early')
    #ax0.plot(time, filter_avg[1][59:(59+92)] - filter_avg[2][59:(59+92)], linewidth=2.5, color='red' , label='late' )
    ax0.plot(time, filter_avg[0][59:(59+92)], linewidth=2.5, color='blue', label='early')
    ax0.plot(time, filter_avg[1][59:(59+92)], linewidth=2.5, color='red' , label='late' )
    #ax0.plot(time, filter_avg[2][59:(59+92)], color='k' ,   label='average')
    #ax0.plot(time, filter_avg[2][59:(59+92)], linewidth=2.5, color='purple', label='early')

    #print(filter_avg[0][59:(59+92)] - filter_avg[2][59:(59+92)])
    #print(filter_avg[0][59:(59+92)])
    #print(filter_avg[2][59:(59+92)])

    # ============ plot the standard deviation data ============
    #ax0.fill_between(time, filter_avg[0][59:(59+92)] - filter_avg[2][59:(59+92)] - 0.75*filter_std[0][59:(59+92)], filter_avg[0][59:(59+92)] - filter_avg[2][59:(59+92)] + 0.75*filter_std[0][59:(59+92)], alpha=0.2)
    #ax0.fill_between(time, filter_avg[1][59:(59+92)] - filter_avg[2][59:(59+92)] - 0.75*filter_std[1][59:(59+92)], filter_avg[1][59:(59+92)] - filter_avg[2][59:(59+92)] + 0.75*filter_std[1][59:(59+92)], alpha=0.2)
    ax0.fill_between(time, filter_avg[0][59:(59+92)] - 0.75*filter_std[0][59:(59+92)], filter_avg[0][59:(59+92)] + 0.75*filter_std[0][59:(59+92)], alpha=0.2)
    ax0.fill_between(time, filter_avg[1][59:(59+92)] - 0.75*filter_std[1][59:(59+92)], filter_avg[1][59:(59+92)] + 0.75*filter_std[1][59:(59+92)], alpha=0.2)
    #ax0.fill_between(time, filter_avg[2][59:(59+92)] - filter_std[2][59:(59+92)], filter_avg[2][59:(59+92)] + filter_std[2][59:(59+92)], alpha=0.2)

    # ============ Set the ticks ===============================
    ax0.tick_params(axis='x', labelsize=16)
    ax0.tick_params(axis='y', labelsize=16)


    ax0.plot([day_notation[0] - 59, day_notation[0] - 59], [-20, 20], 'b--', linewidth=2.5, alpha=0.5,)
    ax0.plot([day_notation[1] - 59 -1, day_notation[1] - 59 -1], [-20, 20], 'r--', linewidth=2.5, alpha=0.5,)

    ax0.legend(loc='upper left', fontsize=20)
    #plt.xticks(rotation=45)

    plt.savefig("/home/sun/paint/phd/phd_C5_fig1c_v0_filtered_tp_timeseries_new_2070.pdf")

def acquire_time():
    import datetime

    # 设置开始日期和结束日期
    start_date = datetime.date(1980, 3, 1)
    end_date = datetime.date(1980, 5, 31)

    # 生成逐日的时间列表并转换为字符串格式
    date_list = []
    current_date = start_date
    while current_date <= end_date:
        date_list.append(current_date.strftime('%d-%b'))
        current_date += datetime.timedelta(days=1)

    return date_list

def band_pass_calculation(data, fs, low_frq, high_frq, order,):
    '''
        fs: sample freq
    '''
    lowcut  = 1/low_frq
    highcut = 1/high_frq

    b, a    = butter(N=order, Wn=[lowcut, highcut], btype='band', fs=fs)

    filtered_data = filtfilt(b, a, data)

    return filtered_data


if __name__ == '__main__':
    early_years, late_years = screen_early_late(onset_day_file_42['onset_day'].data)

    # screen the onset early/late years
    onset_day_file_early = onset_day_file.sel(year=early_years)
    onset_day_file_late  = onset_day_file.sel(year=late_years)

    onset_day_avg_e        = np.average(onset_day_file_early['onset_day'])
    onset_day_avg_l        = np.average(onset_day_file_late['onset_day'])
    

    #print(onset_day_file_late)
    # screen the olr_bandpass result to fit the early/late years for March-April-May
    olr_bandpass_file_early = olr_bandpass_file.sel(time=olr_bandpass_file.time.dt.year.isin(early_years))
    olr_bandpass_file_late  = olr_bandpass_file2.sel(time=olr_bandpass_file2.time.dt.year.isin(late_years))

    olr_early, olr_early_all               = cal_time_evolution_OLR_2edition(olr_bandpass_file_early, "tp_filter")
    olr_late,  olr_late_all                = cal_time_evolution_OLR_2edition(olr_bandpass_file_late, "tp_filter")
    #olr_late[130:150] *= 1.5
    olr_origin,olr_origin_all              = cal_time_evolution_OLR_2edition(olr_bandpass_file, "tp_filter")

    


#    #print(olr_bandpass_file_late)
#    early_high_olr, early_low_olr, late_high_olr, late_low_olr, origin = cal_time_evolution_OLR(olr_bandpass_file_early, olr_bandpass_file_late, 'BOB_olr_10_30', 'BOB_olr_30_70', olr_bandpass_file)
#
#    # Send to plot function
    time_axis = acquire_time()

    frq1 = 8 ; frq2 = 40 ; frq3 = 80
    plot_bandpass_olr([olr_early, olr_late, olr_origin], [olr_early_all, olr_late_all, olr_origin_all], time_axis, [onset_day_avg_e, 137])

   