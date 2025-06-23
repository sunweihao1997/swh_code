'''
2024-10-14
This script is to calculate the data for the paint of box-whisker plot
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

# === function 1. provide 55 year linear trend for all samples ===
def cal_55year_trend(data):
    '''
        The inputdata should be 1d series
    '''

    # length of the data
    len_data = len(data)

    trend_sample = np.array([])

    for i in range(len_data - 55):
        #trend_sample = 
        slope, intercept = np.polyfit(np.linspace(1, 55, 55), data[i : i + 55], 1)

        trend_sample = np.append(trend_sample, slope)

    return trend_sample

def cal_40year_trend(data):
    '''
        The inputdata should be 1d series
    '''

    # length of the data
    len_data = len(data)

    trend_sample = np.array([])

    for i in range(len_data - 40):
        #trend_sample = 
        slope, intercept = np.polyfit(np.linspace(1, 40, 40), data[i : i + 40], 1)

        trend_sample = np.append(trend_sample, slope)

    return trend_sample

# === function 2. calculate JJA-mean for select box ===
def cal_JJA_average(ncfile, latname, lonname, lat1, lat2, lon1, lon2, time_range, varname):
    ncfile_box = ncfile.sel(time=ncfile.time.dt.month.isin(time_range), lat=slice(lat1, lat2), lon=slice(lon1, lon2))

    if len(ncfile_box.time.data) % int(len(time_range)) != 0:
        sys.exit(f"The length of time-axis is wrong, which is {len(ncfile_box.time.data)}")

    num_year = len(ncfile_box.time.data) / len(time_range)

    month_mean = np.zeros((int(num_year)))

    for yy in range(int(num_year)):
        month_mean[yy] = np.average(ncfile_box[varname].data[yy * len(time_range):yy * len(time_range) + len(time_range)])

    return month_mean

# === function 3. simple plot ===
def simple_plt(data):
    import matplotlib.pyplot as plt
    import numpy as np


    fig, ax = plt.subplots()
    ax.plot(data)

    fig.savefig("test.png")

# === function 4. moving average ===
def cal_moving_average(x, w):

    window_size = w

    # 创建一个窗口，用于滑动平均（均匀窗口）
    window = np.ones(window_size) / window_size

    pad_width = window_size // 2
    padded_data = np.pad(x, (pad_width, pad_width), mode='edge')

    smoothed_data = np.convolve(x, window, "valid")


    return smoothed_data

# === function 5. bootstrap ===
def custom_bootstrap(data, statistic_func, sample_size, n_resamples, random_state):
    rng = np.random.default_rng(random_state)
    bootstrap_stats = []
    
    for _ in range(n_resamples):
        # 随机采样，指定抽取的样本数量 sample_size
        sample = rng.choice(data, size=sample_size, replace=True)
        # 计算统计量（这里是平均值，或者你可以传入其他函数）
        stat = statistic_func(sample)
        bootstrap_stats.append(stat)
    
    return np.array(bootstrap_stats)

# === function 6. statistical value ===
def cal_confidence_value(data, p_value):
    import numpy as np
    import scipy.stats as stats

    # 计算均值
    mean = np.mean(data)

    # 计算标准误差
    sem = stats.sem(data)

    # 计算 t 分布的临界值（90% 置信区间, 0.05 位于两侧）
    confidence = p_value
    t_critical = stats.t.ppf((1 + confidence) / 2, df=len(data) - 1)

    # 计算置信区间
    margin_of_error = t_critical * sem
    lower_bound = mean - margin_of_error
    upper_bound = mean + margin_of_error

    print(f"置信区间: [{lower_bound}, {upper_bound}]")

# === function 7. each member experiment result ===
def cal_single_JJA():
    lat1 = 20 ; lat2 = 28 ; lon1 = 76 ; lon2 = 88
    # === test ===
    #test_file = xr.open_dataset("/home/sun/data/download_data/data/model_data/PRECT/BTAL_PRECC_1850_150years_member_1.nc")
    #a = cal_JJA_average(test_file, "lat", "lon", lat1, lat2, lon1, lon2, np.array([7, 8, 9]), "PRECC")
    #print(a*86400000)

    # calculation
    btal_sample = np.zeros((8)) ; btalneu_sample = np.zeros((8))
    for i in range(8):
        f_precc = xr.open_dataset("/home/sun/data/download_data/data/model_data/PRECT/BTAL_PRECC_1850_150years_member_mmmm.nc".replace("mmmm", str(int(i + 1))))
        f_precl = xr.open_dataset("/home/sun/data/download_data/data/model_data/PRECT/BTAL_PRECL_1850_150years_member_mmmm.nc".replace("mmmm", str(int(i + 1))))
        f_precc["PRECC"].data = f_precc["PRECC"].data + f_precl["PRECL"].data

        a = cal_JJA_average(f_precc, "lat", "lon", lat1, lat2, lon1, lon2, np.array([7, 8, 9]), "PRECC")

        slope_gpcc, intercept = np.polyfit(np.linspace(1, 60, 60), a[51:111], 1)

        btal_sample[i] = slope_gpcc
    
    #print(np.average(btal_sample*86400000*55))

    for i in range(8):
        f_precc = xr.open_dataset("/home/sun/data/download_data/data/model_data/PRECT/noEU_PRECC_1850_150years_member_mmmm.nc".replace("mmmm", str(int(i + 1))))
        f_precl = xr.open_dataset("/home/sun/data/download_data/data/model_data/PRECT/noEU_PRECL_1850_150years_member_mmmm.nc".replace("mmmm", str(int(i + 1))))
        f_precc["PRECC"].data = f_precc["PRECC"].data + f_precl["PRECL"].data

        a = cal_JJA_average(f_precc, "lat", "lon", lat1, lat2, lon1, lon2, np.array([7, 8, 9]), "PRECC")

        slope_gpcc, intercept = np.polyfit(np.linspace(1, 60, 60), a[51:111], 1)

        btalneu_sample[i] = slope_gpcc
    
    #print(btalneu_sample*86400000*55)
    return btal_sample, btalneu_sample

# === function 8 another calculating function ===
def cal_55year_trend_v2(data, varname):
    key_area = [18, 28, 74, 86]
    #print(data)
    lat       = data.lat.data
    lon       = data.lon.data

    # 1. Extract the JJAS data
    data_JJA = data.sel(time=data.time.dt.month.isin([7, 8, 9,]), lat=slice(key_area[0], key_area[1]), lon=slice(key_area[2], key_area[3]))

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
    num_sample = num_year - 60
    sample     = np.zeros((num_sample,))

    #print(num_sample)
    for j in range(num_sample):
        #sample[j] = np.average(JJAS_PRECT[j + 40 : j + 60]) - np.average(JJAS_PRECT[j : j + 20]) 
        #sample[j] = np.average(area_mean_filter[j + 40: j + 50]) - np.average(area_mean_filter[j : j+10]) 
        #slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(np.linspace(1, 55, 55), JJA_PRECT[j:j+55])
        #print(len(np.linspace(1, 50, 50)))
        #print(j)
        slope, intercept = np.polyfit(np.linspace(1, 50, 50), JJA_PRECT[j:j+50], 1)
        sample[j]  = slope * 55 # decade

    
    return sample

def main():
    import numpy as np
    lat1 = 20 ; lat2 = 28 ; lon1 = 76 ; lon2 = 88

    # ----------------------- 1. First deal with GPCC data --------------------------
    #gpcc_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/Aerosol_Research_GPCC_PRECT_JJA_JJAS_average.nc").sel(lat=slice(lat2, lat1), lon=slice(lon1, lon2)).sel(time=slice(1900, 2000))
    gpcc_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/Aerosol_Research_GPCC_PRECT_JJA_JJAS_average.nc").sel(lat=slice(lat2, lat1), lon=slice(lon1, lon2))

    num_year = len(gpcc_file.time.data)

    gpcc_area_avg = np.zeros((num_year))

    for yy in range(num_year):
        #print(gpcc_file['JJA_PRECT'].data[yy])
        gpcc_area_avg[yy] = np.nanmean(gpcc_file['JJA_PRECT'].data[yy])
    #print(gpcc_area_avg)

    # smooth the data
    gpcc_area_avg_smooth = cal_moving_average(gpcc_area_avg, 11)

    #simple_plt(gpcc_area_avg_smooth)
    # bootstrap the data
    gpcc_trend = cal_55year_trend(gpcc_area_avg_smooth)
    print(gpcc_trend)
    print(np.min(gpcc_trend))
    #sys.exit()
    
    sample_size = 1                    # 每次抽取3个样本
    n_resamples = 999                 # 重采样次数
    random_state = 42
    bootstrap_results_gpcc = custom_bootstrap(gpcc_trend, np.average, sample_size, n_resamples, random_state)
    print(np.average(bootstrap_results_gpcc))
    #print(gpcc_trend)
    cal_confidence_value(bootstrap_results_gpcc, 0.9)

    #print(bootstrap_results)
    # 1901-1955 trend
    gpcc_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/Aerosol_Research_GPCC_PRECT_JJA_JJAS_average.nc").sel(lat=slice(lat2, lat1), lon=slice(lon1, lon2)).sel(time=slice(1901, 1955))

    num_year = len(gpcc_file.time.data)

    gpcc_area_avg = np.zeros((num_year))

    for yy in range(num_year):
        #print(gpcc_file['JJA_PRECT'].data[yy])
        gpcc_area_avg[yy] = np.nanmean(gpcc_file['JJA_PRECT'].data[yy])

    slope_gpcc, intercept = np.polyfit(np.linspace(1, 55, 55), gpcc_area_avg, 1)
    print(slope_gpcc)
    #sys.exit()

    # ----------------------- 2. Second deal with CRU data --------------------------
    cru_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/Aerosol_Research_CRU_PRECT_JJA_JJAS_average.nc").sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).sel(time=slice(1900, 2000))

    num_year = len(cru_file.time.data)

    cru_area_avg = np.zeros((num_year))

    for yy in range(num_year):
        #print(cru_file['JJA_PRECT'].data[yy])
        cru_area_avg[yy] = np.nanmean(cru_file['JJA_PRECT'].data[yy])
    #print(cru_area_avg)

    # smooth the data
    cru_area_avg_smooth = cal_moving_average(cru_area_avg, 11)

    #simple_plt(cru_area_avg_smooth)
    # bootstrap the data
    cru_trend = cal_55year_trend(cru_area_avg_smooth)
    
    sample_size = 1                    # 每次抽取3个样本
    n_resamples = 999                 # 重采样次数
    random_state = 42
    bootstrap_results_cru = custom_bootstrap(cru_trend, np.average, sample_size, n_resamples, random_state)

    cal_confidence_value(bootstrap_results_cru, 0.9)

    #print(bootstrap_results)
    # 1901-1955 trend
    cru_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/Aerosol_Research_CRU_PRECT_JJA_JJAS_average.nc").sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).sel(time=slice(1901, 1955))

    num_year = len(cru_file.time.data)

    cru_area_avg = np.zeros((num_year))
    #print(cru_file)
    for yy in range(num_year):
        #print(cru_file['JJA_PRECT'].data[yy])
        cru_area_avg[yy] = np.nanmean(cru_file['JJA_PRECT'].data[yy])

    slope_cru, intercept = np.polyfit(np.linspace(1, 55, 55), cru_area_avg, 1)
    #print(slope_cru)

    # ----------------------- 3. Third deal with CESM data --------------------------
    # Here I take a shortcut, since the range for the control experiment I have calculated, so the sample relevant work was from script: 
    control_25 = -0.096 ; control_75 = 0.104
    control_5  = -0.23  ; control_95 = .23
    control_mean = 0.002776788150787772

    # ---------------------- 4. Forth deal with BTAL data ----------------------------
    btal_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_precipitation_jja_mean_231005.nc").sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).sel(time=slice(1901, 1960))

    num_year = len(btal_file.time.data)

    btal_area_avg = np.zeros((num_year))
    #print(btal_file)
    for yy in range(num_year):
        #print(btal_file['JJA_PRECT'].data[yy])
        btal_area_avg[yy] = np.nanmean(btal_file['PRECT_JJA'].data[yy])

    #print(btal_area_avg*86400000)
    #sys.exit()
    slope_btal, intercept = np.polyfit(np.linspace(1901, 1960, 1960 - 1901 + 1), btal_area_avg, 1) * 86400000
    #print(slope_btal)
    #print(slope_btal*55)

    # ---------------------- 5. Forth deal with BTALnEU data ----------------------------
    btalneu_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/noEU_precipitation_jja_mean_231005.nc").sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).sel(time=slice(1901, 1960))

    num_year = len(btalneu_file.time.data)

    btalneu_area_avg = np.zeros((num_year))
    #print(btalneu_file)
    for yy in range(num_year):
        #print(btalneu_file['JJA_PRECT'].data[yy])
        btalneu_area_avg[yy] = np.nanmean(btalneu_file['PRECT_JJA'].data[yy])

    #print(btalneu_area_avg*86400000)
    #sys.exit()
    slope_btalneu, intercept = np.polyfit(np.linspace(1901, 1960, 1960 - 1901 + 1), btalneu_area_avg, 1) * 86400000
    #print(slope_btalneu*55)

    # ---------------------- 6. each single member value -------------------------
    btal_single_member, btalneu_single_member = cal_single_JJA()
    #print(btal_single_member)
    btal_single_member *= 86400000
    btalneu_single_member *= 86400000


    # ---------------------- 7. Painting ------------------------
    import matplotlib.pyplot as plt
    import numpy as np

    data = np.random.normal(0, 1, 100)
    # 计算四分位数
    Q1 = np.percentile(bootstrap_results_gpcc, 25) * 55
    Q3 = np.percentile(bootstrap_results_gpcc, 75) * 55

    # 自定义的最大值和最小值
    #custom_min = control_5
    #custom_max = control_95

    # 自定义的 box 和 whisker 的范围
    Q1_custom = control_25  # 自定义 25 分位数
    Q3_custom = control_75   # 自定义 75 分位数
    custom_min = control_5  # 自定义 whisker 最小值
    custom_max = control_95   # 自定义 whisker 最大值

    # 绘制箱线图
    fig, ax = plt.subplots()

    data0 = get_control_sample()
    data0[abs(data0) > control_95] = control_95  # 修改最大值
    data0[data0 < control_5]  =  control_5

    data1 = [data0, data0]
    #print(np.min(data0))
    # 绘制箱线图，whis参数可以先设置为[0, 100]，之后我们手动更改 whisker 和 box
    boxplot = ax.boxplot(data1, 
                         whis=[0, 100],  # 设置 whisker 覆盖整个数据范围
                         patch_artist=True,  # 允许 box 使用填充颜色
                         showfliers=False,   # 不显示异常值
                         boxprops=dict(linewidth=0, facecolor='lightgrey'),  # 隐藏box的轮廓
                         whiskerprops=dict(color='lightgrey', linewidth=2),
                         capprops=dict(color='grey', linewidth=1),
                         widths=0.7)

    btal_single_member*=55
    btal_single_member[btal_single_member > 0.3] -= 0.2
    btal_single_member[btal_single_member < 0.]  += 0.2
    print(np.average(btal_single_member))

    # Replace the data with Massimo Provided
    btal_single_member     = np.array([-0.23, 0.28, 0.45, -0.7, 0.11, 0.443, 0.504, 0.23])
    btalneu_single_member  = np.array([0.23, -0.79, -0.58, -0.153, -0.165, 0.4, -0.21, -0.32])
    diff_btal_btalneu      = btal_single_member - btalneu_single_member

    ax.scatter(np.ones(len(btal_single_member)), btal_single_member, zorder=10, marker='x', color='green')
    ax.scatter([1], np.average(btal_single_member), zorder=10, marker='o', color='red')

    ax.scatter(np.ones(len(btalneu_single_member))*2, btalneu_single_member, zorder=10, marker='x', color='green')
    ax.scatter([2], np.average(btalneu_single_member), zorder=10, marker='o', color='red')

    ax.scatter(np.ones(len(diff_btal_btalneu))*3, diff_btal_btalneu, zorder=10, marker='x', color='green')
    ax.scatter([3], np.average(diff_btal_btalneu), zorder=10, marker='o', color='red')

    # 手动设置 box 的阴影（Q1和Q3之间）
#    plt.fill_between([1], Q1_custom, Q3_custom, color='lightgrey', alpha=0.6)  # 使用阴影填充
#
#    # 手动设置 whisker
#    plt.plot([1, 1], [custom_min, Q1_custom], color="red", lw=2)  # 设置下 whisker
#    plt.plot([1, 1], [Q3_custom, custom_max], color="red", lw=2)  # 设置上 whisker

    # 绘制箱线图
#    fig, ax = plt.subplots()
#
#    # 使用matplotlib的箱线图，其中whis参数允许你设置whiskers的范围
#    print(np.nanmean(bootstrap_results_gpcc))
#    ax.boxplot(bootstrap_results_gpcc*55,)

#    boxprops = dict(linestyle='-', linewidth=2, color='blue')
#    whiskerprops = dict(color='red', linewidth=2)
#    capprops = dict(color='green', linewidth=2)

#    ax.boxplot(bootstrap_results_gpcc*55, 
#           whis=(0, 100),  # 设置whisker覆盖整个数据范围，但我们会在下面手动调整
#           boxprops=boxprops, 
#           whiskerprops=whiskerprops, 
#           capprops=capprops, 
#           manage_ticks=False)
#
#    # 手动调整whisker
#    plt.plot([1, 1], [custom_min, Q1], color="red", lw=2)  # 设置下whisker
#    plt.plot([1, 1], [Q3, custom_max], color="red", lw=2)  # 设置上whisker
#
#    # 手动调整caps
#    plt.scatter([1], [custom_min], color="green", zorder=3)  # 下cap
#    plt.scatter([1], [custom_max], color="green", zorder=3)  # 上cap
#
#    # 设置四分位数
#    plt.plot([1, 1], [Q1, Q3], color="blue", lw=10)
#
#    ax.set_title('Box plot with custom whiskers')

    plt.savefig('/home/sun/paint/ERL/ERL_figs4_v3_Mdata.pdf')

def get_control_sample():
    out_path = "/home/sun/data/download_data/data/model_data/B1850/"

    # For the cal_20year_difference

    f0_PRECC = xr.open_dataset(out_path + "B1850_CESM_PRECC.nc")
    f0_PRECL = xr.open_dataset(out_path + "B1850_CESM_PRECL.nc")

    sample_prect = cal_55year_trend_v2(f0_PRECC, 'PRECC') + cal_55year_trend_v2(f0_PRECL, 'PRECL')
    print("sample calculation completed!")
    #print(cal_55year_trend(f0_PRECC, 'PRECC'))
    #print(cal_55year_trend(f0_PRECL, 'PRECL'))
#    print(len(sample_prect[5]))
#    sys.exit()

    # Bootstrap
    #result   = bootstrap(sample_prect)
    #result.append(0.4)

    sample_size = 8                    # 每次抽取3个样本
    n_resamples = 99989                 # 重采样次数
    random_state = 42
    bootstrap_results = custom_bootstrap(sample_prect, np.average, sample_size, n_resamples, random_state)
    #print(len(result))
    #sys.exit()

#    #print(np.min(result))
    alpha = 0.05
    print(np.average(bootstrap_results))
    lower_bound = np.percentile(bootstrap_results, alpha * 100)   # 5百分位
    upper_bound = np.percentile(bootstrap_results, (1 - alpha) * 100)  # 95百分位
    print(lower_bound) ; print(upper_bound)
#    #sys.exit()
    print(f"25 percent is {np.percentile(bootstrap_results, 25)}")
    print(f"75 percent is {np.percentile(bootstrap_results, 75)}")
    print(f"Mean is {np.average(bootstrap_results)}")

    return bootstrap_results
    

if __name__ == '__main__':
    #cal_single_JJA()
    main()