'''
2023-12-18
This script is to deal with the data from GPCC, to show the influence of the aerosol on the long-term change in precipitation over Indian continent
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import scipy

# ========================= File Location =================================

f0       = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc").sel(time=slice(1900, 2000))
#print(f0)

#print(f0)
time     = f0.time.data
lat      = f0.lat.data # 90 to -90
lon      = f0.lon.data

years    = len(f0.time.data)
print('Years length is {}'.format(years))
year     = f0.time.data
print(year)

# =========================================================================

# ======================== Function for smoothing =========================

def cal_moving_average(x, w):
        return np.convolve(x, np.ones(w), "valid") / w

# =========================================================================

# ======================== Paint for area-averaged mean evolution ================

def plot_area_average_evolution(f0, varname, extent):
    '''
        extent: latmin, latmax, lonmin, linmax
    '''
    # 1. Extract data from original file using extent

    f0_area = f0.sel(lat=slice(extent[0], extent[1]), lon=slice(extent[2], extent[3]))

    # 2. Claim the array for saving the 165-year data

    area_mean = np.zeros((years))

    # 3. Calculate for each year

    for yy in range(years):
        area_mean[yy] = np.average(f0_area[varname].data[yy])

    # 4. Smoothing
    N = 3
    period = 20
    Wn = 2 * (1 / period) / 1

    b, a = scipy.signal.butter(N, Wn, 'lowpass')

    area_mean_filter = scipy.signal.filtfilt(b, a, area_mean, axis=0)

    print('The length after filter is {}'.format(len(area_mean_filter)))

#    print('The length of area_mean is {}'.format(len(area_mean_smooth)))
    # 5. Paint
    fig, ax = plt.subplots()

   #ax.plot(year, area_mean - np.average(area_mean), color='grey', linewidth=1.5)

#    time_process = np.linspace(1891 + (w-1)/2,  2006 - (w-1)/2, int((2006 - (w-1)/2) - (1891 + (w-1)/2) + 1))

    ax.plot(year, area_mean_filter - np.average(area_mean_filter), color='orange', linewidth=2.5)
#    ax.plot(area_mean_smooth, color='orange', linewidth=2.5)

    ax.set_ylim((-0.5, 0.5 ))
    
    ax.set_ylabel("BTAL", fontsize=11)

    ax.set_title("CESM", loc='left', fontsize=15)
    ax.set_title("74-86°E, 18-28°N", loc='right', fontsize=15)

    #plt.savefig("/home/sun/paint/EUI_GPCC_PRECT_evolution_area_average_Indian_key_region.pdf", dpi=700)
    plt.savefig("test1.png", dpi=700)

    #print(len(area_mean_smooth))

    #print(area_mean - np.average(area_mean))




def main():
    #f0 = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/CESM_BTAL_esemble_JJAS_precipitation.nc').sel(time=slice(1900, 2000))
    plot_area_average_evolution(f0, 'PRECT_JJA_BTAL', [18, 28, 74, 86])
    #print(JJAS_mean)


if __name__ == '__main__':
    main()