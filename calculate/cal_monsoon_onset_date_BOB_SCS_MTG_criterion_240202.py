'''
2024-2-2
This script is to calculate the monsoon onset dates using MTG criterion for the BOB and SCS area

The dataset includes: ERA5
'''

import xarray as xr
import numpy as np
import os

# ==== File location information ====

src_path = '/home/sun/ERA5_data/temperature/ERA5_temperature_extraction_500_200/'

# ===================================

# ==== Function : Identify the completeness of the data ====

def verify_number_for_each_year(src_path):
    full_list = os.listdir(src_path) ; full_list.sort()

    for yyyy in range(1940, 2024):
        year_list = [element for element in full_list if 'temperature_'+str(yyyy) in element]
        if len(year_list) != 365 and len(year_list) != 366:
            print(f'It is year {yyyy}, the number of files is {len(year_list)}')

# ==== Function : Identify the onset dates for the BOB monsoon ====

def BOB_monsoon_onset_time_series():
    '''
        This function is to calculate the BOBSM onset dates using U850 criterion
    '''
    # ---- 1. File list process ----

    full_list = os.listdir(src_path) ; full_list.sort()

    # --- 2. Get the 365-length area-averaged U850 ---
    # The array includes total 83 years everyday area-averaged uwind at 850 hPa
    BOB_MTG_total = np.zeros((83, 365))

    extent = [5, 15, 90, 100]

    j = 0
    for yyyy in range(1940, 2023):
        # The array of the 365-length U850
        BOB_MTG = np.array([]) 

        year_list = [element for element in full_list if 'temperature_'+str(yyyy) in element]

        year_list.sort()

        for ffff in year_list:
            f0 = xr.open_dataset(src_path + ffff)

            BOB_MTG = np.append(BOB_MTG, BOB_monsoon_onset_criterion_MTG(f0, extent))

        BOB_MTG_total[j, :] = BOB_MTG[:365]

        j += 1
    
    print(f'MTG time series over BOB has completed')

    return BOB_MTG_total


def BOB_monsoon_onset_criterion_MTG(f0, extent):
    '''
        This function is to calculate the BOBSM onset dates using U850 criterion
    '''

    # ---- 1. Select the area data ----

    f0_BOB = f0.sel(longitude=slice(extent[2], extent[3]), level=slice(300, 500))
    #print(f0_BOB)
    print((np.nanmean(f0_BOB['t'].data[0, :, 0, :]) - np.nanmean(f0_BOB['t'].data[0, :, 1, :])))

    return (np.nanmean(f0_BOB['t'].data[0, :, 0, :]) - np.nanmean(f0_BOB['t'].data[0, :, 1, :]))

def BOB_monsoon_u850_test(series):
    import matplotlib.pyplot as plt

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }

    x = np.linspace(1, 365, 365)
    y = series - np.average(series)

    plt.plot(x, y, 'k')
    plt.plot([1, 365], [0, 0], 'r--')
    plt.plot([1, 365], [3, 3], 'g--')
    plt.title('BOBSM U850 Time Series', loc='left', fontdict=font)
    plt.title('5 to 15, 85 to 100', loc='right', fontdict=font)

    plt.xlabel('Daily', fontdict=font)
    plt.ylabel('Deviation', fontdict=font)

    plt.grid(True)
    plt.xticks(np.linspace(10, 360, 36), rotation='vertical')


    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('/home/sun/paint/onset_day/ERA5_BOB_U850_time_series3.pdf')

def judge_the_onset_date(series):
    '''
        This function uses the following criterions:
            1. onset_day to onset_day +5 the averaged index greater than 0
            2. in the subsequent 20 days, the BOBSM index be positive in at least 15 days
            3. cumulative 20day mean greater than 1
    '''

    start_day = 85 # I need not to calculate from the first day

    for dd in range(start_day, start_day + 120):
        if series[dd] < 0:
            continue
        elif np.average(series[dd : dd + 5]) < 0:  # First estimation
            continue
        elif np.sum(series[dd : dd + 20] > 0) < 15: # Second estimation
            continue
        elif np.sum(series[dd : dd + 20])/20 < 1:  # Third estimation
            continue

        else:
            break

    return dd + 1

def plot_onset_dates(series):
    import matplotlib.pyplot as plt
    import scipy.signal

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }

    x  = np.linspace(1940, 2022, 83)
    y1 = series

    # Low-pass filter
    N = 3

    period = 10

    Wn = 2 * (1 / period) / 1

    b, a = scipy.signal.butter(N, Wn, 'lowpass')

    y2   = scipy.signal.filtfilt(b, a, y1 ,axis = 0) 

    plt.plot(x, y1, 'grey', linewidth=0.8)
    plt.plot(x, y2, 'orange', linewidth=1.5)
    plt.plot([1940, 2022], [np.average(series), np.average(series)], 'r--')
    plt.title('ISM onset dates (U850)', loc='left', fontdict=font)
    plt.title('5 to 15, 40 to 80', loc='right', fontdict=font)

    plt.xlabel('Year', fontdict=font)
    plt.ylabel('Days', fontdict=font)

    plt.grid(True)
    plt.xticks(np.linspace(1940, 2020, 9),)


    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('/home/sun/paint/onset_day/ERA5_Indian_U850_onset_dates_10year_filter.pdf')

def spectral_analysis(time_series):
    from numpy .fft import fft , ifft
    import matplotlib.pyplot as plt
    import scipy.signal

    date_fft = np.fft.fft(time_series - np.mean(time_series))

    date_squared =  (( np.sqrt( np.real(date_fft)**2 + np.imag(date_fft)**2))[0:40]**2)/100

    f_year   = np.linspace(0, 39, 40) / len(time_series)

    tau      = len(time_series) / np.linspace(0, 39, 40)

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }

    x  = tau
    y1 = date_squared

    plt.plot(x, y1, 'grey', linewidth=1.2)

    #plt.title('BOBSM onset dates (U850)', loc='left', fontdict=font)
    #plt.title('5 to 15, 85 to 100', loc='right', fontdict=font)

    plt.xlabel('Cycles per year', fontdict=font)
    plt.ylabel('Spectral Power', fontdict=font)

    plt.grid(True)
    #plt.xticks(np.linspace(1940, 2020, 9),)


    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('/home/sun/paint/onset_day/ERA5_BOB_U850_onset_dates_spectral_power.pdf')


# ==== Function : Identify the onset dates for the SCS monsoon ====
def SCS_monsoon_onset_time_series():
    '''
        This function is to calculate the SCSSM onset dates using U850 criterion
    '''
    # ---- 1. File list process ----

    full_list = os.listdir(src_path) ; full_list.sort()

    # --- 2. Get the 365-length area-averaged U850 ---
    # The array includes total 83 years everyday area-averaged uwind at 850 hPa
    SCS_u850_total = np.zeros((83, 365))

    extent = [5, 15, 110, 120]

    j = 0
    for yyyy in range(1940, 2023):
        # The array of the 365-length U850
        SCS_u850 = np.array([]) 

        year_list = [element for element in full_list if 'wind_'+str(yyyy) in element]

        year_list.sort()

        for ffff in year_list:
            f0 = xr.open_dataset(src_path + ffff)

            SCS_u850 = np.append(SCS_u850, BOB_monsoon_onset_criterion_u850(f0, extent))

        SCS_u850_total[j, :] = SCS_u850[:365]

        j += 1
    
    print(f'U850 time series over SCS has completed')

    return SCS_u850_total

def SCS_monsoon_u850_test(series):
    import matplotlib.pyplot as plt

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }

    x = np.linspace(1, 365, 365)
    y = series - np.average(series)

    plt.plot(x, y, 'k')
    plt.plot([1, 365], [0, 0], 'r--')
    plt.plot([1, 365], [3, 3], 'g--')
    plt.title('SCSSM U850 Time Series', loc='left', fontdict=font)
    plt.title('5 to 15, 110 to 120', loc='right', fontdict=font)

    plt.xlabel('Daily', fontdict=font)
    plt.ylabel('Deviation', fontdict=font)

    plt.grid(True)
    plt.xticks(np.linspace(10, 360, 36), rotation='vertical')


    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('/home/sun/paint/onset_day/ERA5_SCS_U850_time_series.pdf')

# ==== Function : Identify the onset dates for the Indian monsoon ====
def Indian_monsoon_onset_time_series():
    '''
        This function is to calculate the Indian SM onset dates using U850 criterion
    '''
    # ---- 1. File list process ----

    full_list = os.listdir(src_path) ; full_list.sort()

    # --- 2. Get the 365-length area-averaged U850 ---
    # The array includes total 83 years everyday area-averaged uwind at 850 hPa
    Indian_u850_total = np.zeros((83, 365))

    extent = [5, 15, 40, 80]

    j = 0
    for yyyy in range(1940, 2023):
        # The array of the 365-length U850
        Indian_u850 = np.array([]) 

        year_list = [element for element in full_list if 'wind_'+str(yyyy) in element]

        year_list.sort()

        for ffff in year_list:
            f0 = xr.open_dataset(src_path + ffff)

            Indian_u850 = np.append(Indian_u850, BOB_monsoon_onset_criterion_u850(f0, extent))

        Indian_u850_total[j, :] = Indian_u850[:365]

        j += 1
    
    print(f'U850 time series over Indian has completed')

    return Indian_u850_total

def Indian_monsoon_u850_test(series):
    import matplotlib.pyplot as plt

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }

    x = np.linspace(1, 365, 365)
    y = series - np.average(series)

    plt.plot(x, y, 'k')
    plt.plot([1, 365], [0, 0], 'r--')
    plt.plot([1, 365], [6.2, 6.2], 'g--')
    plt.title('IndianSM U850 Time Series', loc='left', fontdict=font)
    plt.title('5 to 15, 40 to 80', loc='right', fontdict=font)

    plt.xlabel('Daily', fontdict=font)
    plt.ylabel('Deviation', fontdict=font)

    plt.grid(True)
    plt.xticks(np.linspace(10, 360, 36), rotation='vertical')


    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('/home/sun/paint/onset_day/ERA5_Indian_U850_time_series.pdf')

def judge_the_onset_date_ISM(series):
    '''
        This function uses the following criterions:
            1. onset_day to onset_day +5 the averaged index greater than 0
            2. in the subsequent 20 days, the BOBSM index be positive in at least 15 days
            3. cumulative 20day mean greater than 1
    '''

    start_day = 90 # I need not to calculate from the first day

    for dd in range(start_day, start_day + 200):
        if series[dd] < 6:
            continue
        elif np.average(series[dd : dd + 6]) < 6:  # First estimation
            continue

        else:
            break

    return dd + 1




def main():
    # 1. ==== Identify the completeness of the data ====
    verify_number_for_each_year(src_path) # !! It is OK !!

    # 2. ==== Deal with the BOB summer monsoon ====
    time_series_MTG_BOB = BOB_monsoon_onset_time_series()

    # Save the result to the file
    out_path = '/home/sun/data/onset_day_data/'
    #np.save(out_path + 'BOB_u850_time_series_83years_select_area3.npy', time_series_u850)

    # 2.1 plot the climatology
    #time_series_u850 = np.load(out_path + 'BOB_u850_time_series_83years_select_area3.npy')
    #BOB_monsoon_u850_test(np.average(time_series_u850, axis=0))

    # 2.2 Get onset date for each year
    #onset_date = np.array([])
    #for i in range(time_series_u850.shape[0]):
    #    onset_date = np.append(onset_date, judge_the_onset_date(time_series_u850[i]))

    #print(np.average(onset_date))

    # 2.3 Plot the onset dates from 1940 to 2022
    #plot_onset_dates(onset_date)

    # 3. ==== Spectral Analysis ====
    #spectral_analysis(onset_date)

    # 4. ==== Deal with the SCS summer monsoon ====
    #time_series_u850 = SCS_monsoon_onset_time_series()

    # 4.1 Save the result to the file
#    out_path = '/home/sun/data/onset_day_data/'
#    #np.save(out_path + 'SCS_u850_time_series_83years_select_area.npy', time_series_u850)
#
#    # 4.2 plot the climatology
#    time_series_u850 = np.load(out_path + 'SCS_u850_time_series_83years_select_area.npy')
#    #SCS_monsoon_u850_test(np.average(time_series_u850, axis=0))
#
#    # 4.3 Get onset date for each year
#    onset_date = np.array([])
#    for i in range(time_series_u850.shape[0]):
#        onset_date = np.append(onset_date, judge_the_onset_date(time_series_u850[i]))
#
#    # 4.4 Plot the onset dates from 1940 to 2022
#    plot_onset_dates(onset_date)
#
#     # 5. ==== Deal with the Indian summer monsoon ====
#    #time_series_u850 = Indian_monsoon_onset_time_series()
#
#    # 5.1 Save the result to the file
#    out_path = '/home/sun/data/onset_day_data/'
#    #np.save(out_path + 'Indian_u850_time_series_83years_select_area.npy', time_series_u850)
#
#    # 5.2 plot the climatology
#    time_series_u850 = np.load(out_path + 'Indian_u850_time_series_83years_select_area.npy')
#    #Indian_monsoon_u850_test(np.average(time_series_u850, axis=0))
#
#    # 5.3 Get onset date for each year
#    onset_date = np.array([])
#    for i in range(time_series_u850.shape[0]):
#        onset_date = np.append(onset_date, judge_the_onset_date_ISM(time_series_u850[i]))
#
#    # 5.4 Plot the onset dates from 1940 to 2022
#    plot_onset_dates(onset_date)

    

if __name__ == '__main__':
    main()