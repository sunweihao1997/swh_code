'''
2023-10-5
This script is to cdo cat dispersive data files into on file per ensemble member
'''
import os

path0 = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/BTAL/'
path1 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/SLP/'

for i in range(8):
    path3 = path0 + 'BTAL_' + str(i + 1) +'/mon/atm/PSL/'

    #print(os.listdir(path3))
    os.system('cdo cat ' + path3 + '*nc ' + path1 + 'BTAL_SLP_1850_150years_member_' + str(i + 1) + '.nc')
    #os.system('ncdump -h ' + path1 + 'noEU_SLP_1850_150years_member_' + str(i + 1) + '.nc')