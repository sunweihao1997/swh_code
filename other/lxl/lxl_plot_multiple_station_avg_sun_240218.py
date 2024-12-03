import xarray as xr
import numpy as np
import os

path = "/home/sun/data/liuxl_sailing/post_process/quanzhougang/"

def plot_multiple_pic(ws10m, title_left, title_right, pic_name):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(figsize=(30, 10))

    #axs.plot(np.linspace(0, 71, 72), ws10m[0:72, 1], color='red', linewidth=3)
    axs.plot(np.linspace(0, 71, 72), ws10m[0:72, 0], linewidth=3, label='CMA', color='red')
    axs.plot(np.linspace(0, 71, 72), ws10m[0:72, 1], linewidth=3, label='IFS', color='black')
    axs.plot(np.linspace(0, 71, 72), ws10m[0:72, 2], linewidth=3, label='GFS', color='blue')
    axs.plot(np.linspace(0, 71, 72), ws10m[0:72, 3], linewidth=3, label='SD3', color='purple')
    

    # Set the attribution for axis
    axs.set_xticks(range(0, 71, 4))
    axs.set_xticklabels(range(0, 71, 4), fontsize=25)

    axs.set_yticks(range(0, 60, 10))
    axs.set_yticklabels(range(0, 60, 10), fontsize=25)

    # Set the Titile
    axs.set_title(title_left, loc='left', fontsize=25)
    axs.set_title(title_right, loc='right', fontsize=25)

    #plt.savefig('/home/sun/paint/lxl_paint/{}.png'.format(pic_name))

    # This command will make the picture to show the legend
    axs.legend()

    plt.savefig('{}.png'.format(pic_name))

def calculate_nearst_15_average(datasets_name, date_name, file_name, var_name):
    file_path = datasets_name + '/'
    time_path = date_name + '/'

    f0        = xr.open_dataset(path + file_path + time_path + file_name)

    ws10m       = f0[var_name].data

    return np.average(ws10m, axis=1)

datasets_name = ['CMA', 'IFS', 'GFS', 'SD3']
#datasets_name = ['CMA']

#path1         = path + datasets_name[0] + '/'

date0         = '2023072612'

# Sun add: Calculate the 15station average for 4 situation
ws10m_obs_model = np.zeros((240, 4)) # For the second dimension, the first is observation while the other three is models
ws10m_obs_model[:, 0] = calculate_nearst_15_average(datasets_name=datasets_name[0], date_name=date0, file_name='ws10m.nc', var_name= "ws10m")
ws10m_obs_model[:, 1] = calculate_nearst_15_average(datasets_name=datasets_name[1], date_name=date0, file_name='ws10m.nc', var_name= "ws10m")
ws10m_obs_model[:, 2] = calculate_nearst_15_average(datasets_name=datasets_name[2], date_name=date0, file_name='ws10m.nc', var_name= "ws10m")
ws10m_obs_model[:, 3] = calculate_nearst_15_average(datasets_name=datasets_name[3], date_name=date0, file_name='ws10m.nc', var_name= "ws10m")

plot_multiple_pic(ws10m_obs_model, date0, 'WS10m', 'sun_model_comparison_ws10m')