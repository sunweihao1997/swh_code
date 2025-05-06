'''
2024-9-26
This script is deal with the data from CESM 1850 Pi-Control experiment

The purpose is to provide dataset for bootstrap
'''
import xarray as xr
import numpy as np
import sys
import os
import scipy
#from cdo import *

sys.path.append('/home/sun/uoe-code/module/')
from module_sun import cal_xydistance

# ============================ File Location ==================================

path_src = "/exports/csce/datastore/geos/groups/aerosol/sabineCESM/B1850AL/mon/atm/"

var_name = ['PRECC', 'PRECL']

key_area = [18, 28, 74, 86]

# =============================================================================

def cal_moving_average(x, w):
        return np.convolve(x, np.ones(w), "same") / w

# ============================ CDO the data ===================================

def cdo_cat(path_src, out_path, out_name):

    cdo = Cdo()

    # Get the list
    file_list = os.listdir(path_src) ; file_list.sort()

    cdo.cat(input = [(path_src + x) for x in file_list], output = out_path + out_name)

# =============================================================================

# ===================== Calculate the 20year difference        ================

def cal_55year_trend(data, varname):
    #print(data)
    lat       = data.lat.data
    lon       = data.lon.data

    # 1. Extract the JJAS data
    data_JJAS = data.sel(time=data.time.dt.month.isin([6, 7, 8,]), lat=slice(key_area[0], key_area[1]), lon=slice(key_area[2], key_area[3]))

    # 2. Claim the array to save the result
    num_year = int(len(data_JJAS.time.data)/4)
    #print(num_year) # total 202 year

    JJAS_PRECT = np.zeros((num_year,))
    
    # 3. Calculation
    for i in range(num_year):
        JJAS_PRECT[i] = np.average(data_JJAS[varname].data[i*4 : i*4+4])
    JJAS_PRECT = cal_moving_average(JJAS_PRECT, 11) * 86400000
#    # 4. Smoothing
#    N = 2   
#    period = 5
#    Wn = 2 * (1 / period) / 1
#
#    b, a = scipy.signal.butter(N, Wn, 'lowpass')

    area_mean_filter = JJAS_PRECT

    # 4. Calculate sample
    num_sample = num_year - 57
    sample     = np.zeros((num_sample,))

    for j in range(num_sample):
        #sample[j] = np.average(JJAS_PRECT[j + 40 : j + 60]) - np.average(JJAS_PRECT[j : j + 20]) 
        #sample[j] = np.average(area_mean_filter[j + 40: j + 50]) - np.average(area_mean_filter[j : j+10]) 
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.linspace(1, 50, 50), JJAS_PRECT[j:j+50])
        sample[j]  = slope * 50 # decade

    
    return sample

# =============================================================================

# ========================== Bootstrap ========================================

def bootstrap(data):
    from scipy.stats import bootstrap
    rng = np.random.default_rng()

    data = (data,)

    res  = bootstrap(data, np.average, batch=1, n_resamples=9999, confidence_level=0.90, random_state=rng)

    return res

# =============================================================================

def get_pdf(data, mu, sigma):

    x = data

    y = scipy.stats.norm.pdf(data, mu, sigma)

    return x, y

def plot_pdf(data,labelx,labely,title,color,out_path,out_name,low,high,adjustment):
    """Plot ecdf"""
    import matplotlib.pyplot as plt
    # Call get_ecdf function and assign the returning values
    #data = np.append(data, 0.2)
    

    data += adjustment ; low += adjustment ; high += adjustment
    mu   = np.average(data)
    sigma= np.std(data)
    x, y = get_pdf(data, mu, sigma)
    y/=100

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    

    
    #plt.plot(x,y,marker='.',linestyle='none',c='k')
    plt.plot(x_sorted,y_sorted,c='k')

    plt.xlabel(labelx)
    plt.ylabel(labely)
    plt.title(title)

    plt.xlim((-0.03, 0.03))
    plt.ylim((0, 1))

    plt.xticks([-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03,])
    plt.yticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100])
    
#    plt.fill_between(x, y, color='lightblue', alpha=0.5)
    #plt.fill_between([np.max(x)-0.04, 0.4], [1, 1], color='lightblue', alpha=0.5)

#    plt.plot([-0.7, low], [0.025, 0.025], color='r')
#    plt.plot([high, 0.7], [0.975, 0.975], color='r')
#
#    plt.plot([low, low], [0., 0.025], color='r')
#    plt.plot([high,high],   [0., 0.975], color='r')


    plt.savefig(out_path + out_name, dpi=650)


# ========================== main =============================================

def main():
    # For the cdo_cat

    out_path = "/home/sun/data/download_data/data/model_data/B1850/"

    # For the cal_20year_difference

    f0_PRECC = xr.open_dataset(out_path + "B1850_CESM_PRECC.nc")
    f0_PRECL = xr.open_dataset(out_path + "B1850_CESM_PRECL.nc")

    sample_prect = cal_55year_trend(f0_PRECC, 'PRECC') + cal_55year_trend(f0_PRECL, 'PRECL')
    print("sample calculation completed!")
    print(sample_prect)
    #sys.exit()

    # Bootstrap
    result   = bootstrap(sample_prect)
    #result.append(0.4)

    print(result.confidence_interval) #-0.28 to 0.15, at 95% confidence
    print(np.max(result.bootstrap_distribution))
    print(np.min(result.bootstrap_distribution))

    # CDF plot
    path_src = '/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/model_evaluation/'
    plot_pdf(result.bootstrap_distribution," ", " ", "PDF", "r", path_src, "Aerosol_Research_CESM_PRECT_40years_trend_Bootstrap_PDF.pdf", result.confidence_interval[0], result.confidence_interval[1], 0.005)



if __name__ == '__main__': 
    main()