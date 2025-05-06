'''
2023-10-5
This script is to cdo cat dispersive data files into on file per ensemble member
'''
import os

path_src1 = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/BTAL/'
path_src2 = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/BTALnEU/'

path1 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/FSDS/'

for i in range(8):
    path3 = path_src1 + 'BTAL_'    + str(i + 1) +'/mon/atm/FSDS/'
    path4 = path_src2 + 'BTALnEU_' + str(i + 1) +'/mon/atm/FSDS/'

    os.system('cdo cat ' + path3 + '*nc ' + path1 + 'BTAL_FSDS_1850_150years_member_' + str(i + 1) + '.nc')
    os.system('cdo cat ' + path4 + '*nc ' + path1 + 'BTALnEU_FSDS_1850_150years_member_' + str(i + 1) + '.nc')