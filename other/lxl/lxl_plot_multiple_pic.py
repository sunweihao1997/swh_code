import xarray as xr
import numpy as np
import os

path = "/home/sun/data/liuxl_sailing/post_process/quanzhougang/"

def plot_multiple_pic(datasets_name, date_name, file_name, var_name, title_left, title_right, pic_name):
    file_path = datasets_name + '/'
    #time_path = '2023072612/' # UTC 
    time_path = date_name + '/'
    #vari_path = "ws10m.nc'
    #print(path+file_path) #/home/sun/data/liuxl_sailing/post_process/quanzhougang/CMA/

    f0        = xr.open_dataset(path + file_path + time_path + file_name)

    #print(f0)
    #num_station = f0['num_station'].data
    #station_name= f0['station_name'].data
    ws10m       = f0[var_name].data
    print(ws10m.shape)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(figsize=(30, 10))

    #axs.plot(np.linspace(0, 71, 72), ws10m[0:72, 1], color='red', linewidth=3)
    axs.plot(np.linspace(0, 71, 72), ws10m[0:72], linewidth=1.5)

    # Set the attribution for axis
    axs.set_xticks(range(0, 71, 4))
    axs.set_xticklabels(range(0, 71, 4), fontsize=25)

    axs.set_yticks(range(0, 60, 10))
    axs.set_yticklabels(range(0, 60, 10), fontsize=25)

    # Set the Titile
    axs.set_title(title_left, loc='left', fontsize=25)
    axs.set_title(title_right, loc='right', fontsize=25)

    #plt.savefig('/home/sun/paint/lxl_paint/{}.png'.format(pic_name))
    plt.savefig('{}.png'.format(pic_name))
    
datasets_name = ['CMA', 'IFS', 'GFS', 'SD3']
#datasets_name = ['CMA']

#path1         = path + datasets_name[0] + '/'

date0         = '2023072612'
for i in datasets_name:
    #path_dataset = '/home/sun/data/liuxl_sailing/post_process/quanzhougang/' + i + date0 + '/'

    plot_multiple_pic(datasets_name=i, date_name=date0, file_name='ws10m.nc', var_name= "ws10m",  title_left = i, title_right='quanzhou', pic_name= i+"ws10m")