'''
2024-2-8
This script is to calculate the monsoon onset dates for the BOBSM and SCSSM using CESM2_pacemaker experiment

This goal is to evaluate the simulation of the CESM2 on monsoon onset
'''
import xarray as xr
import numpy as np
import os
#from cdo import *

# ==== File location information ====

src_path = '/home/sun/data/download_data/CESM2_pacemaker/u850/daily/'

# ===================================

# ==== File list process ====
# Under this path, the python script and BSSP370 should be removed

file_list0 = os.listdir(src_path)
file_list  = []

for ffff in file_list0:
    if "python" in ffff or "BSSP" in ffff:
        continue
    else:
        file_list.append(ffff)

file_list.sort()

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ==== Function: filter out the member =======

def return_same_member_file(list_all, keyword):
    '''
        This function is used to return a list including files for each member
    '''
    new_list = []
    for ffff in list_all:
        if keyword in ffff:
            new_list.append(ffff)
        else:
            continue

    print(f"The list including {keyword} has been fitered, which includes {len(new_list)}")

    new_list.sort()

    return new_list



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ==== Function: CDO cat for each member ====

def cdo_cat_files(end_path, list_files, member_str):

    cdo = Cdo()

    list_files.sort()

    cdo.cat(input=[(src_path + x) for x in list_files], output=end_path + "CESM2.pacemaker.u850." + member_str + ".188001-201412.nc")

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ==== Function: Verify the integrity of the cated data ====

def verify_time_integrity(start, end, path, file):
    '''
        This function read and export numbers for each year, if one year includes less than 365 files, print relevant information
    '''
    f_input = xr.open_dataset(path + file)
    # print(len(f_input.time.data)) # totally 49276 days, 135 * 365 + 1 (2015.1.1)

    for yy in range(start, end + 1):
        f_input_1yr = f_input.sel(time=f_input.time.dt.year.isin([yy]))

        if len(f_input_1yr.time.data) != 365:
            print(f'It is the year {yy}, the containing time length is {len(f_input_1yr.time.data)}')

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ==== Function: BOB Monsoon Onset ====
def BOB_monsoon_onset_time_series(f0):
    '''
        This function is to calculate the BOBSM onset dates using U850 criterion
    '''

    # --- 1. Get the 365-length area-averaged U850 ---
    # The array includes total 135 years everyday area-averaged uwind at 850 hPa
    BOB_u850_total = np.zeros((135, 365)) # 135 yr's daily data

    extent = [5, 15, 90, 100]

    j = 0
    for yy in range(1880, 2014 + 1):
        # The array of the 365-length U850
        BOB_u850 = np.array([]) 

        f0_year  = f0.sel(time=f0.time.dt.year.isin([yy])).sel(lat=slice(extent[0], extent[1]), lon=slice(extent[2], extent[3]))

        if len(f0_year.time.data) != 365:
            print(f'It is the year {yy}, which yr time length is not 365, it is {len(f0_year.time.data)}')

        for dd in range(365):
            BOB_u850 = np.append(BOB_u850, np.nanmean(f0_year['U850'].data[dd]))

        BOB_u850_total[j, :] = BOB_u850

        j += 1
    
    #print(f'U850 time series over BOB has completed')

    return BOB_u850_total

def BOB_monsoon_u850_test(series, max_line, min_line):
    import matplotlib.pyplot as plt

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }

    x = np.linspace(1, 365, 365)
    y0 = np.average(series, axis=0)   - np.average(np.average(series, axis=0))
    y1 = max_line - np.average(series, axis=0)
    y2 = min_line - np.average(series, axis=0)

    fig, ax = plt.subplots()



    ax.plot(x, y0, 'k')
    ax.fill_between(x, y2 + y0, y1 + y0, alpha=0.2)

    ax.plot([1, 365], [0, 0], 'r--')

    plt.title('BOBSM U850 Time Series', loc='left', fontdict=font)
    plt.title('5 to 15, 90 to 100', loc='right', fontdict=font)

    plt.xlabel('Daily', fontdict=font)
    plt.ylabel('Deviation', fontdict=font)

    plt.grid(True)
    plt.xticks(np.linspace(10, 360, 36), rotation='vertical')


    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('/home/sun/paint/onset_day/CESM2_pacemaker_BOB_U850_time_series_interannual_membermean.pdf')

# ==== Function: CSC Monsoon Onset ====
def SCS_monsoon_onset_time_series(f0):
    '''
        This function is to calculate the BOBSM onset dates using U850 criterion
    '''

    # --- 1. Get the 365-length area-averaged U850 ---
    # The array includes total 135 years everyday area-averaged uwind at 850 hPa
    BOB_u850_total = np.zeros((135, 365)) # 135 yr's daily data

    extent = [5, 15, 110, 120]

    j = 0
    for yy in range(1880, 2014 + 1):
        # The array of the 365-length U850
        BOB_u850 = np.array([]) 

        f0_year  = f0.sel(time=f0.time.dt.year.isin([yy])).sel(lat=slice(extent[0], extent[1]), lon=slice(extent[2], extent[3]))

        if len(f0_year.time.data) != 365:
            print(f'It is the year {yy}, which yr time length is not 365, it is {len(f0_year.time.data)}')

        for dd in range(365):
            BOB_u850 = np.append(BOB_u850, np.nanmean(f0_year['U850'].data[dd]))

        BOB_u850_total[j, :] = BOB_u850

        j += 1
    
    #print(f'U850 time series over BOB has completed')

    return BOB_u850_total

def SCS_monsoon_u850_test(series, max_line, min_line):
    import matplotlib.pyplot as plt

    font = {'family': 'serif',
            'color':  'darkred',
            'weight': 'normal',
            'size': 12,
            }

    x = np.linspace(1, 365, 365)
    y0 = np.average(series, axis=0)   - np.average(np.average(series, axis=0))
    y1 = max_line - np.average(series, axis=0)
    y2 = min_line - np.average(series, axis=0)

    fig, ax = plt.subplots()



    ax.plot(x, y0, 'k')
    ax.fill_between(x, y2 + y0, y1 + y0, alpha=0.2)

    ax.plot([1, 365], [0, 0], 'r--')

    plt.title('BOBSM U850 Time Series', loc='left', fontdict=font)
    plt.title('5 to 15, 110 to 120', loc='right', fontdict=font)

    plt.xlabel('Daily', fontdict=font)
    plt.ylabel('Deviation', fontdict=font)

    plt.grid(True)
    plt.xticks(np.linspace(10, 360, 36), rotation='vertical')


    # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.savefig('/home/sun/paint/onset_day/CESM2_pacemaker_SCS_U850_time_series_interannual_membermean.pdf')

def main():
    # 1. Verify whether the files is completed, and cdo cat 1880-2014 to new path
    end_path = '/home/sun/data/download_data/CESM2_pacemaker/u850/daily_cdo/'

#    for i in range(1, 11):
#        if i<10:
#            list_member = return_same_member_file(file_list, "pacific.00"+ str(i) + ".cam.")
#            member_str  = "00" + str(i)
#            cdo_cat_files(end_path, list_member, member_str)
#
#        else:
#            list_member = return_same_member_file(file_list, "pacific.0"+ str(i) + ".cam.")
#            member_str  = "0" + str(i)
#            cdo_cat_files(end_path, list_member, member_str)

    # 2. Verify the time integrity for each year
    list_new = os.listdir(end_path) ; list_new.sort()

    #for single_file in list_new[0:
    #verify_time_integrity(1880, 2015, end_path, list_new[0]) # !! Pass the check !!

    # 3. Start to calculate the onset dates using U850 criterion

    # 3.1 Claim the array which save the 10members daily BOB-area averaged U850
    BOB_10m_u850 = np.zeros((10, 135, 365)) # 10 members, 135 years, 365 days
    for i in range(10):
        f_member        = xr.open_dataset(end_path + list_new[i])

        BOB_10m_u850[i] = BOB_monsoon_onset_time_series(f_member)

        print(f'Sucessfully calculate U850 for the whole period for member {i+1}')
    
    # 3.2 Save U850 time-series to the file
    ncfile  =  xr.Dataset(
        {
            "BOB_u850":    (["member", "year", "day"], BOB_10m_u850),
        },
        coords={
            "member": (["member"], np.linspace(1, 10, 10)),
            "year":   (["year"],   np.linspace(1880, 2014, 135)),
            "day":    (["day"],    np.linspace(1, 365, 365)),
        },
            )

    ncfile.attrs['description'] = 'Created on 2024-2-8 on the Huaibei Server, script name is cal_CESM2_pacemaker_BOB_SCS_monsoon_onset_dates_evaluation_240208.py. This file calculate daily value of the area-averaged U850 over BOB, given the period 1880-2014 for the 10 members. Areas 5-15, 90-100'

    ncfile.to_netcdf('/home/sun/data/process/analysis/model_simulation_monsoon_onset/CESM2_pacemaker_10members_1880-2014_BOB_U850_timeseries.nc')

    f0_BOB_u850 = xr.open_dataset('/home/sun/data/process/analysis/model_simulation_monsoon_onset/CESM2_pacemaker_10members_1880-2014_BOB_U850_timeseries.nc')

    le_cli_u850 = np.average(f0_BOB_u850['BOB_u850'].data[:, :, :], axis=0)

    print(le_cli_u850.shape)

    max_u850    = np.array([])
    min_u850    = np.array([])

    for i in range(365):
        max_u850 = np.append(max_u850, np.max(le_cli_u850[:, i]))
        min_u850 = np.append(min_u850, np.min(le_cli_u850[:, i]))

    # 4. Plot the climatology evolution of the BOB U850

    BOB_monsoon_u850_test(le_cli_u850, max_u850, min_u850)

# 5. Start to calculate the onset dates using U850 criterion

    # 3.1 Claim the array which save the 10members daily BOB-area averaged U850
#    SCS_10m_u850 = np.zeros((10, 135, 365)) # 10 members, 135 years, 365 days
#    for i in range(10):
#        f_member        = xr.open_dataset(end_path + list_new[i])
#
#        SCS_10m_u850[i] = SCS_monsoon_onset_time_series(f_member)
#
#        print(f'Sucessfully calculate U850 for the whole period for member {i+1}')
#    
#    # 3.2 Save U850 time-series to the file
#    ncfile  =  xr.Dataset(
#        {
#            "SCS_u850":    (["member", "year", "day"], SCS_10m_u850),
#        },
#        coords={
#            "member": (["member"], np.linspace(1, 10, 10)),
#            "year":   (["year"],   np.linspace(1880, 2014, 135)),
#            "day":    (["day"],    np.linspace(1, 365, 365)),
#        },
#            )
#
#    ncfile.attrs['description'] = 'Created on 2024-2-8 on the Huaibei Server, script name is cal_CESM2_pacemaker_BOB_SCS_monsoon_onset_dates_evaluation_240208.py. This file calculate daily value of the area-averaged U850 over SCS, given the period 1880-2014 for the 10 members'
#
#    ncfile.to_netcdf('/home/sun/data/process/analysis/model_simulation_monsoon_onset/CESM2_pacemaker_10members_1880-2014_SCS_U850_timeseries.nc')

#    f0_SCS_u850 = xr.open_dataset('/home/sun/data/process/analysis/model_simulation_monsoon_onset/CESM2_pacemaker_10members_1880-2014_SCS_U850_timeseries.nc')
#
#    le_cli_u850 = np.average(f0_SCS_u850['SCS_u850'].data[:, :, :], axis=0)
#
#    print(le_cli_u850.shape)
#
#    max_u850    = np.array([])
#    min_u850    = np.array([])
#
#    for i in range(365):
#        max_u850 = np.append(max_u850, np.max(le_cli_u850[:, i]))
#        min_u850 = np.append(min_u850, np.min(le_cli_u850[:, i]))
#
#    # 4. Plot the climatology evolution of the SCS U850
#
#    SCS_monsoon_u850_test(le_cli_u850, max_u850, min_u850)


if __name__ == '__main__':
    main()