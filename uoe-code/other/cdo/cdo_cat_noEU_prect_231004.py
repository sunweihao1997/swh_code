'''
2023-10-4
This script is to cdo cat dispersive data files into on file per ensemble member
'''
import os

path0 = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/BTALnEU/'
path1 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/PRECT/'

for i in range(8):
    path3 = path0 + 'BTALnEU_' + str(i + 1) +'/mon/atm/PRECC/'
    path4 = path0 + 'BTALnEU_' + str(i + 1) +'/mon/atm/PRECL/'

    os.system('cdo cat ' + path3 + '*nc ' + path1 + 'noEU_PRECC_1850_150years_member_' + str(i + 1) + '.nc')
    os.system('cdo cat ' + path4 + '*nc ' + path1 + 'noEU_PRECL_1850_150years_member_' + str(i + 1) + '.nc')