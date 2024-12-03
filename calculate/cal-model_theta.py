'''
2020/12/22
使用模式层的资料来计算位温
时间跨度3-7月
生成的层数：
'''
import os
import numpy as np
import Ngl, Nio
import json
from geopy.distance import distance
import numpy.ma as ma
import sys
import math
from netCDF4 import Dataset
sys.path.append("/data5/2019swh/mycode/module/")
from module_sun import *
from module_writenc import *
from attribute import *

a_pl = {'longname': 'pressure_level','units': 'Pa','valid_range': [-1000000000000000.0, 1000000000000000.0]}

path = '/data1/MERRA2/daily/modellev/'
path3  ='/data1/other_data/DataUpdate/ERA5/merra2/model_theta/'
year = np.arange(1980,1981)

for yyyy in year:
    path2 = path + str(yyyy) +'/'
    file_list = os.listdir(path2)
    file_list.sort()
    #根据平年闰年确定4月1日在第几个
    if leap_year(yyyy):
        location = 60
    else:
        location = 59
    location_end = (location+1)

    for mmmm in range(location,location_end):
        f = Nio.open_file(path2 + file_list[mmmm])
        lev = f.variables["lev"][:]
        lon = f.variables["lon"][:]
        lat = f.variables["lat"][:]
        pl = f.variables["PL"][:]
        t = f.variables["T"][:]
        theta  =  model_theta(t,pl)
        u   = f.variables["U"][:]
        v   = f.variables["V"][:]

        time = np.arange(0,1)+1
        file = create_nc_multiple(path3, "model_theta_"+str(yyyy)+out_date(yyyy,mmmm+1), time, lev, lon, lat)
        add_variables(file, "theta", theta, a_T, 1)
        add_variables(file, "PL", pl,a_pl,1)
        add_variables(file,"uwind",u,a_uwind,1)
        add_variables(file,"vwind",v,a_vwind,1)

        file.close()

