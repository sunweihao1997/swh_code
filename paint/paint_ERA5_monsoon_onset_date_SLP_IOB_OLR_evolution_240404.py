'''
2024-4-4
This script is to plot the evolution of IOB/SLP/OLR index during the onset normal/early/date years
'''
import xarray as xr
import numpy as np

onset_date_file = xr.open_dataset('/home/sun/data/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc').sel(year=slice(1980, 2020))
SLP_file        = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_SLP_month_land_slp_70-90_1940-2020.nc').sel(year=slice(1980, 2020))
OLR_file        = xr.open_dataset('/home/sun/data/process/ERA5/ERA5_OLR_month_maritime_continent_1940-2020.nc').sel(year=slice(1980, 2020))
iob_file       = xr.open_dataset('/home/sun/data/process/HadISST/HadISST_SST_IOB_month_1940_2020.nc').sel(year=slice(1980, 2020))

# ==== Filter out onset early/normal/late years ====
onset_date      = onset_date_file['onset_day'].data
#print(np.average(onset_date))
onset_early     = np.array([])
onset_late      = np.array([])

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

def quick_test(array1, array2, array3, left, right, figname):
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

    print(array3_min)

    # Data for plotting
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.linspace(1, 15, 15), array1, color='k', label='climate')
    ax.plot(np.linspace(1, 15, 15), np.average(array2, axis=0), color='b', label='onset early')
    ax.plot(np.linspace(1, 15, 15), np.average(array3, axis=0), color='r', label='onset late')

    ax.fill_between(np.linspace(1, 15, 15), np.average(array2, axis=0) - 0.7 * np.std(array2), np.average(array2, axis=0) + 0.7 * np.std(array2), color='b', alpha=0.2)
    ax.fill_between(np.linspace(1, 15, 15), np.average(array3, axis=0) - 0.7 * np.std(array3), np.average(array3, axis=0) + 0.7 * np.std(array3), color='r', alpha=0.2)

    xticks  = np.linspace(1, 15, 15)
    xlabels = ['Oct(-1)', 'Nov(-1)', 'Dec(-1)', 'Jan', 'Feb', 'Mar', 'Apr', "May", 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels)

    ax.set_title(left, loc='left', fontsize=17.5)
    ax.set_title(right, loc='right', fontsize=17.5)

    ax.legend()
    ax.grid()

    fig.savefig('/home/sun/paint/ERA5/monsoon_onset_abnormal/' + figname)

# ============= Calculate Climatology ================
# Last year Nov-Dec + current year 12 months
slp_climate = np.zeros((15))
olr_climate = np.zeros((15))
iob_climate = np.zeros((15))
slp_climate_year = np.average(SLP_file['FMA_SLP'].data, axis=0)
olr_climate_year = np.average(OLR_file['FMA_OLR'].data, axis=0)
iob_climate_year = np.average(iob_file['month_iob'].data, axis=0)

slp_climate[0:3] = slp_climate_year[-3:]
olr_climate[0:3] = olr_climate_year[-3:]
slp_climate[3:]  = slp_climate_year
olr_climate[3:]  = olr_climate_year
iob_climate[0:3] = iob_climate_year[-3:]
iob_climate[3:]  = iob_climate_year


slp_early   = np.zeros((len(onset_early), 15))
slp_late    = np.zeros((len(onset_late),  15))
olr_early   = np.zeros((len(onset_early), 15))
olr_late    = np.zeros((len(onset_late),  15))
iob_early   = np.zeros((len(onset_early), 15))
iob_late    = np.zeros((len(onset_late),  15))

j = 0 ; k = 0
for yyyy in range(1980, 2021):
    if yyyy in onset_early:
        slp_early[j][3:] = SLP_file['FMA_SLP'].data[yyyy-1980]
        slp_early[j][0:3] = SLP_file['FMA_SLP'].data[yyyy-1980 -1 ][-3:]
        olr_early[j][3:] =  OLR_file['FMA_OLR'].data[yyyy-1980]
        olr_early[j][0:3] = OLR_file['FMA_OLR'].data[yyyy-1980 -1 ][-3:]
        iob_early[j][3:] =  iob_file['month_iob'].data[yyyy-1980]
        iob_early[j][0:3] = iob_file['month_iob'].data[yyyy-1980 -1 ][-3:]

        j += 1
    elif yyyy in onset_late:
        slp_late[k][3:] = SLP_file['FMA_SLP'].data[yyyy-1980]
        slp_late[k][0:3] = SLP_file['FMA_SLP'].data[yyyy-1980 -1][-3:]
        olr_late[k][3:] =  OLR_file['FMA_OLR'].data[yyyy-1980]
        olr_late[k][0:3] = OLR_file['FMA_OLR'].data[yyyy-1980 -1 ][-3:]
        iob_late[k][3:] =  iob_file['month_iob'].data[yyyy-1980]
        iob_late[k][0:3] = iob_file['month_iob'].data[yyyy-1980 -1 ][-3:]

        k += 1

    

#quick_test(slp_climate, np.average(slp_early, axis=0), np.average(slp_late, axis=0))
#print(olr_early.shape)
quick_test(olr_climate, olr_early, olr_late, 'Maritime Continent OLR', '(-5-10N, 100-130E)', 'ERA5_Maritime_continent_OLR_month_evolution_normal_abnormal_year.png')
quick_test(slp_climate, slp_early, slp_late, 'Indian Peninsula SLP', '(5-20N, 70-90E)', 'ERA5_Indian_continent_SLP_month_evolution_normal_abnormal_year.png')
quick_test(iob_climate, iob_early, iob_late, 'IOB SST', '(-20-20N, 40-100E)', 'ERA5_IOB_SST_month_evolution_normal_abnormal_year.png')