'''
2024-11-1
This script is to calculate the onset date of BOBSM using precipitation criteria
'''
import sys
sys.path.append("/home/sun/mycode/paint")
import paint_lunwen_version3_0_fig1_bob_onset_seris as plv3_1
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import paint_lunwen_version3_0_fig2a_tem_gradient_20220426 as plv3_2a
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl
module_path = ["/home/sun/mycode/module/","/data5/2019swh/mycode/module/"]
sys.path.append(module_path[0])
from module_sun import *
from scipy.ndimage import gaussian_filter
from scipy.fft import fft, ifft

data_path = "/home/sun/mydown/ERA5/era5_precipitation_daily/"

# Get file list
data_list = os.listdir(data_path) ; data_list = data_list[41:]
print(data_list)

# --- get the information ---
f0        = xr.open_dataset(data_path + data_list[5])
#print(np.average(f0['tp'].data)) # Unit should be mm/day

# =============================== Function Area ===================================
def judge_monsoon_onset(pr_series, start=90, threshold=5, thresholdday=10):
    '''
        The input is series of precipitation, start from the 90th day
    '''
    for i in range(0, 220):
        if np.average(pr_series[start + i - 2 : start + i + 3]) - np.average(pr_series[0:30]) < threshold :
            continue
        

        else:
            break

    return start + i

def harmonics_sum_f(x, num):
    precipitation_fft = fft(x)
    #print(precipitation_fft.shape)

    n = len(x)
    harmonics_sum = np.zeros(n, dtype=complex)

    harmonics_sum[1:1 + num] = precipitation_fft[1:1 + num]  # first 12 harmonics
    harmonics_sum[-num:] = precipitation_fft[-num:]  

    reconstructed_signal = ifft(harmonics_sum).real

    return reconstructed_signal + np.average(x)

def calculate_area_mean(ncfile, extent):
    ncfile_region = ncfile.sel(latitude=slice(extent[0], extent[1]), longitude=slice(extent[2], extent[3]))

    time_series   = np.average(np.average(ncfile_region['tp'].data, axis=1), axis=1) * 1e3

    return time_series

# ====================================================================================

def main():
    # Claim the empty array to save the onset date
    onset_date_array = np.zeros((42)) # 1980-2021

    extent           = [15, 5, 90, 100]

    for i in range(42):
        f_oneyear = xr.open_dataset(data_path + data_list[i])

        pr_series = calculate_area_mean(f_oneyear, extent)
        #print(pr_series.shape)

        pr_series_smooth = harmonics_sum_f(np.convolve(pr_series, np.ones(5), "same") / 5, 12)

        onset_date_array[i] = judge_monsoon_onset(pr_series_smooth, threshold=5)

    print(np.average(onset_date_array))

    # Write to netcdf-file
    ncfile  =  xr.Dataset(
    {
        "onset_day": (["year"], onset_date_array),
    },
    coords={
        "year": (["year"], np.linspace(1980, 2021, 42)),
    },
    )
    ncfile.attrs['description']  =  'This is onset date calculated by the ERA5 data, using Bing Wang precipitation criteria.'
    ncfile.to_netcdf("/home/sun/data/monsoon_onset_anomaly_analysis/onset_day_data/ERA5_onset_day_Binwang_criteria.nc")

if __name__ == '__main__':
    main()

