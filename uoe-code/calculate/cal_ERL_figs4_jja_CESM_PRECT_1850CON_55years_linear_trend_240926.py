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
import matplotlib.pyplot as plt
#from cdo import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np


# 如果你想使用核密度估计
from scipy.stats import gaussian_kde

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
    data_JJA = data.sel(time=data.time.dt.month.isin([6, 7, 8,]), lat=slice(key_area[0], key_area[1]), lon=slice(key_area[2], key_area[3]))

    # 2. Claim the array to save the result
    num_year = int(len(data_JJA.time.data)/3)
    print(f"Control experiment totally has {num_year} years")
    #print(num_year) # total 202 year

    JJA_PRECT = np.zeros((num_year,))
    
    # 3. Calculation
    for i in range(num_year):
        JJA_PRECT[i] = np.average(data_JJA[varname].data[i*3 : i*3+3])
    JJA_PRECT = cal_moving_average(JJA_PRECT, 11)* 86400000

#    print(JJA_PRECT)
#    # 4. Smoothing
#    N = 2   
#    period = 5
#    Wn = 2 * (1 / period) / 1
#
#    b, a = scipy.signal.butter(N, Wn, 'lowpass')

    area_mean_filter = JJA_PRECT

    # 4. Calculate sample
    num_sample = num_year - 57
    sample     = np.zeros((num_sample,))

    for j in range(num_sample):
        #sample[j] = np.average(JJAS_PRECT[j + 40 : j + 60]) - np.average(JJAS_PRECT[j : j + 20]) 
        #sample[j] = np.average(area_mean_filter[j + 40: j + 50]) - np.average(area_mean_filter[j : j+10]) 
        #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.linspace(1, 55, 55), JJA_PRECT[j:j+55])
        slope, intercept = np.polyfit(np.linspace(1, 50, 50), JJA_PRECT[j:j+50], 1)
        sample[j]  = slope * 50 # decade

    
    return sample

# =============================================================================

# ========================== Bootstrap ========================================

def bootstrap(data):
    from scipy.stats import bootstrap
    rng = np.random.default_rng()

    data = (data,)

    res  = bootstrap(data, np.average, batch=8, n_resamples=99999, confidence_level=0.95, random_state=rng)

    return res

def bootstrap_resample_combined(data, n_samples=1000):
    """
    使用bootstrap方法从原始数据中重采样，并将所有样本合并。
    
    参数：
    data: 原始数据数组
    n_samples: 重采样次数
    
    返回：
    所有重采样样本合并后的数组
    """
    n = len(data)
    # 存储每次重采样后的数据
    resampled_data = []

    for _ in range(n_samples):
        # 从原始数据中进行有放回抽样
        resample = np.random.choice(data, size=n, replace=True)
        resampled_data.append(resample)
    
    # 将所有重采样样本合并为一个大数组
    combined_resampled_data = np.concatenate(resampled_data)
    
    return combined_resampled_data

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
    print(cal_55year_trend(f0_PRECC, 'PRECC'))
    print(cal_55year_trend(f0_PRECL, 'PRECL'))
#    print(len(sample_prect[5]))
#    sys.exit()

    # Bootstrap
    #result   = bootstrap(sample_prect)
    #result.append(0.4)
    result = bootstrap_resample_combined(sample_prect, n_samples=10000)
    #print(len(result))
    #sys.exit()

#    #print(np.min(result))
    alpha = 0.05
    print(np.average(sample_prect))
    lower_bound = np.percentile(sample_prect, alpha * 100)   # 2.5百分位
    upper_bound = np.percentile(sample_prect, (1 - alpha) * 100)  # 97.5百分位
    print(lower_bound) ; print(upper_bound)
#    #sys.exit()
#
#    #result_sort = np.sort(result)


    # 给定的参数
    
    q5 = -0.58
    q95 = 0.42
    mean = -0.006

    # 计算标准差
    sigma = (q95 - q5) / 3.29

    # 生成正态分布数据
    data = np.random.normal(loc=mean, scale=sigma, size=99999)

    data_sorted = np.sort(data)
    cumulative_prob = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    # 绘制CDF
    plt.plot(data_sorted, cumulative_prob*100, marker='none', linestyle='-', label='Empirical CDF')
    plt.xlabel('Data Values')
    plt.ylabel('Cumulative Probability (%)')
    plt.title('Cumulative probability (%)')

    plt.xlim((-1.1, 1.1))

    plt.yticks(np.linspace(0, 100, 11))
    plt.xticks(np.linspace(-1, 1, 11))
    plt.ylim((0, 100))
    #plt.grid(True)
    #plt.legend()
    plt.savefig('test1.png')




if __name__ == '__main__': 
    main()