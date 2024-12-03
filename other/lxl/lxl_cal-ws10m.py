import xarray as xr
import numpy as np
import os
import math
path = "/home/sun/data/liuxl_sailing/post_process/"
#f0   = xr.open_dataset(path + file_path + time_path + file_name)

# 获取当前目录
#current_dir = os.getcwd()

# 列出当前目录下的所有文件
portnames = os.listdir(path)
#print(portnames)

datasets_names = ['IFS', 'GFS', 'SD3']
#datasets_names = ['GFS', 'SD3']
for iport in portnames:
    #portpath = path+iport ++ "/"
    #print(portpath)      # /home/sun/data/liuxl_sailing/post_process/yangjianggang/

    for idataset in datasets_names:
        yyyymmdd = os.listdir(path + iport + "/" + idataset + "/")
        #print(yyyymmdd) #'2023082700',
        #后续把yyymmdd排序
        for iday in  yyyymmdd:
            print(iday)
            tmp_path = path + iport + "/" + idataset + "/"  +iday + "/"
            #print(path + iport + "/" + idataset + "/"  +iday + "/")
            fu   = xr.open_dataset(tmp_path +'u10m.nc')
            fv   = xr.open_dataset(tmp_path +'v10m.nc')
            u10m = fu['u10m'].data
            v10m = fv['v10m'].data

            ws10m = np.zeros(u10m.shape)
            ws10m = (u10m**2 + v10m**2)**0.5
            #print(ws10m)
            # 3. Save the correlation to the array
            ncfile  =  xr.Dataset(
                    {
                        'ws10m': (["time", "num_station"], ws10m),
                    },
                    coords={
                        "time": (["time"], fu.time.data),
                        "num_station": (["num_station"], fu.num_station.data),
                    },
                    )
            ncfile.attrs['description'] = 'created on 2024-2-15. This file cal ws using u and v'

            ncfile.to_netcdf(tmp_path +'ws10m.nc')
