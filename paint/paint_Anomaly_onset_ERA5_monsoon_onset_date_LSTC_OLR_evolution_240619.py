'''
2024-4-4
This script is to plot the evolution of IOB/SLP/OLR index during the onset normal/early/date years
'''
import xarray as xr
import numpy as np

onset_date_file = xr.open_dataset('/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc').sel(year=slice(1980, 2021))

index_file      = xr.open_dataset("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_OLR.nc")

# ==== Filter out onset early/normal/late years ====
onset_date      = onset_date_file['onset_day'].data
#print(np.average(onset_date))
onset_early     = np.array([])
onset_late      = np.array([])

fast_year       = np.array([1997, 1995, 1988, 2012, 2009, 2006, 2002])
slow_year       = np.array([1985, 1984, 1983, 2010, 2008, 1998, 1996])

onset_std       = np.std(onset_date)
#print(onset_std)
for yyyy in range(len(onset_date_file.year.data)):
    if abs(onset_date[yyyy] - np.average(onset_date)) > onset_std:
        if (onset_date[yyyy] - np.average(onset_date)) > 0:
            onset_late = np.append(onset_late, onset_date_file.year.data[yyyy])
        else:
            onset_early = np.append(onset_early, onset_date_file.year.data[yyyy])

    else:
        continue

'''
late: [1983. 1987. 1993. 1997. 2010. 2016. 2018. 2020.]
early:[1984. 1985. 1999. 2000. 2009. 2017.]
'''

def quick_test(array1, array2, array3, fast, slow, left, right, figname):
    import matplotlib.pyplot as plt
    import numpy as np

    # array2 max/min line
    array2_max = array1.copy()
    array2_min = array1.copy()
    array3_max = array1.copy()
    array3_min = array1.copy()

    for i in range(15):
        array2_min[i] = np.min(array2[:, i])
        array2_max[i] = np.max(array2[:, i])
        array3_min[i] = np.min(array3[:, i])
        array3_max[i] = np.max(array3[:, i])

    #print(array3_min)

    bar_width = 0.35
    positions = range(15)

    # Data for plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    #ax.plot(np.linspace(1, 15, 15), array1, color='k', label='climate')
    ax.bar([p - bar_width/2 for p in positions], np.average(array2, axis=0), width=bar_width, color='lightskyblue', label='onset early', zorder=5)
    ax.bar([p + bar_width/2 for p in positions], np.average(array3, axis=0), width=bar_width, color='lightsalmon', label='onset late',   zorder=6)
    
#    ax.plot(np.linspace(0, 14, 15), np.average(fast, axis=0), 'k--', label='Fast El Nino decaying')
#    ax.plot(np.linspace(0, 14, 15), np.average(slow, axis=0), 'k',   label='Slow El Nino decaying')

    ax.fill_between(np.linspace(0, 14, 15), np.average(array2, axis=0) - 0.7 * np.std(array2), np.average(array2, axis=0) + 0.7 * np.std(array2), color='lightskyblue', alpha=0.35)
    ax.fill_between(np.linspace(0, 14, 15), np.average(array3, axis=0) - 0.7 * np.std(array3), np.average(array3, axis=0) + 0.7 * np.std(array3), color='lightsalmon',  alpha=0.35)

    xticks  = np.linspace(0, 14, 15)
    xlabels = ['Oct(-1)', 'Nov(-1)', 'Dec(-1)', 'Jan', 'Feb', 'Mar', 'Apr', "May", 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.set_title(left, loc='left', fontsize=17.5)
    ax.set_title(right, loc='right', fontsize=17.5)

    ax.legend()
    ax.grid()

    fig.savefig('/home/sun/paint/phd/phd_c5_fig9_' + figname)

# ============= Calculate Climatology ================
# Last year Nov-Dec + current year 12 months
slp_climate = np.zeros((15))
olr_climate = np.zeros((15))

# data preprocessing converting 504 to 42 * 12
lstc_index = np.zeros((42, 12))
olr_index  = np.zeros((42, 12))
for i in range(42):
    lstc_index[i] = index_file['LSTC_psl_IOB'].data[i*12:i*12+12]
    olr_index[i]  = index_file['OLR_mari_Afri'].data[i*12:i*12+12]


slp_climate_year = np.average(lstc_index, axis=0)
olr_climate_year = np.average(olr_index,  axis=0)

slp_climate[0:3] = slp_climate_year[-3:]
olr_climate[0:3] = olr_climate_year[-3:]
slp_climate[3:]  = slp_climate_year
olr_climate[3:]  = olr_climate_year

slp_early   = np.zeros((len(onset_early), 15))
slp_late    = np.zeros((len(onset_late),  15))
olr_early   = np.zeros((len(onset_early), 15))
olr_late    = np.zeros((len(onset_late),  15))

# means fast decaying
slp_fast    = np.zeros((len(fast_year), 15))
slp_slow    = np.zeros((len(slow_year),  15))
olr_fast    = np.zeros((len(fast_year), 15))
olr_slow    = np.zeros((len(slow_year),  15))


j = 0 ; k = 0 ; m = 0 ; n = 0
for yyyy in np.linspace(1980, 2021, 42, dtype=int):
    if yyyy in onset_early:
        slp_early[j][3:]  = lstc_index[yyyy-1980]
        slp_early[j][0:3] = lstc_index[yyyy-1980 -1 ][-3:]
        olr_early[j][3:]  = olr_index[yyyy-1980]
        olr_early[j][0:3] = olr_index[yyyy-1980-1][-3:]

        j += 1

    if yyyy in onset_late:
        slp_late[k][3:]  = lstc_index[yyyy-1980]
        slp_late[k][0:3] = lstc_index[yyyy-1980 -1][-3:]
        olr_late[k][3:] =  olr_index[yyyy-1980]
        olr_late[k][0:3] = olr_index[yyyy-1980 -1 ][-3:]

        k += 1

    if yyyy in fast_year:
        print('fast')
        slp_fast[m][3:]  = lstc_index[yyyy-1980]
        slp_fast[m][0:3] = lstc_index[yyyy-1980 -1][-3:]
        olr_fast[m][3:]  = olr_index[yyyy-1980]
        olr_fast[m][0:3] = olr_index[yyyy-1980 -1 ][-3:]

        m += 1

    if yyyy in slow_year:
        print('slow')
        slp_slow[n][3:]  = lstc_index[yyyy-1980]
        slp_slow[n][0:3] = lstc_index[yyyy-1980 -1][-3:]
        olr_slow[n][3:]  = olr_index[yyyy-1980]
        olr_slow[n][0:3] = olr_index[yyyy-1980 -1 ][-3:]

        n += 1

#print(slp_fast)
#print(slp_late)

#quick_test(slp_climate, np.average(slp_early, axis=0), np.average(slp_late, axis=0))
#print(olr_early.shape)
quick_test((slp_climate - slp_climate)/slp_climate, (olr_early - olr_climate), (olr_late - olr_climate), (olr_fast - olr_climate), (olr_slow - olr_climate),'Maritime-Africa DIFF OLR', ' ', 'ERA5_Maritime_continent_OLR_month_evolution_normal_abnormal_year.pdf')
#quick_test((slp_climate - slp_climate)/slp_climate, (slp_early - slp_climate), (slp_late - slp_climate), (slp_fast - slp_climate), (slp_slow - slp_climate),'LSTC', ' ', 'ERA5_Indian_continent_SLP_month_evolution_normal_abnormal_year_line.pdf')
