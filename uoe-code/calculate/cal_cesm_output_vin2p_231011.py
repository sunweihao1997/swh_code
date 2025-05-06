'''
2023-10-11
This script interpolate cesm output into pressure level for the BTAL and noEU experiments
'''
import os
import numpy as np

path0  =  '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/BTAL/BTAL_1/mon/atm/'

vars_name = ['Z3', 'U', 'V', 'OMEGA', 'RELHUM', 'T']
#vars_name = ['T']
exps_name = ['BTAL', 'BTALnEU']

pnew      =  np.array([1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600, 550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70, 50])

out_path  =  '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/'

def cesm_vin2p(exp_name, var_name):
    import numpy as np
    import xarray as xr
    import Ngl

    for pp in range(8):
        # 8 members ensemble
        path1 = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/' + exp_name + '/' + exp_name + '_' + str(pp + 1) + '/mon/atm/' + var_name + '/'

        # Read surface pressure data
        path_pressure = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/' + exp_name + '/' + exp_name + '_' + str(pp + 1) + '/mon/atm/' + 'PS' + '/'

        # file name list
        var_list = os.listdir(path1) ; var_list.sort()
        ps_list  = os.listdir(path_pressure) ; ps_list.sort()

        # ==== Calculate the interpolate data ====
        length   = len(var_list)
        for ii in range(length):
            ps_file0   = xr.open_dataset(path_pressure + ps_list[ii])
            v_file0    = xr.open_dataset(path1 + var_list[ii])

            vin2p_data = vin2p_calculate(ps_file=ps_file0, data_file=v_file0, varname=var_name)

            ref_file   = xr.open_dataset(path1 + var_list[ii])

            # Save the data
            ds = xr.Dataset(
                {
                    var_name:(["time", "lev", "lat", "lon"], vin2p_data),
                },

                coords = {
                    "lon":(["lon"],ref_file.lon.data),
                    "lat":(["lat"],ref_file.lat.data),
                    "time":(["time"],ref_file.time.data),
                    "lev":(["lev"],pnew)
                }
            )

            ds[var_name].attrs = ref_file[var_name].attrs 

            ds.attrs['description']  =  'Created on 11/Oct/2023, this is the CESM intepolated output on pressure level. Created by Weihao Sun (sunweihao97@gmail.com)'
            ds.to_netcdf(out_path + var_list[ii])

def vin2p_calculate(ps_file, data_file, varname):
    '''This subroutine uses Ngl to calculate data value'''
    import Ngl
    import numpy as np

    vin2p = Ngl.vinth2p(data_file[varname].data, data_file['hyam'].data, data_file['hybm'].data, pnew, ps_file['PS'].data, 1, data_file['P0'].data / 100, 1, True)

    '''Mask the value under the ground'''
    dim0 = vin2p.shape[0]
    dim1 = vin2p.shape[1] # level
    dim2 = vin2p.shape[2]
    dim3 = vin2p.shape[3]

    vin2p_mask = vin2p.copy()
    for tt in range(dim0):
        for latt in range(dim2):
            for lonn in range(dim3):
                ps0 = ps_file['PS'].data[tt, latt, lonn] / 100

                for levv in range(dim1):
                    if pnew[levv] > ps0:
                        vin2p_mask[tt, levv, latt, lonn] = np.nan
                    else:
                        continue


    return np.float32(vin2p_mask)


# 
def main():
    for i in exps_name:
        for j in vars_name:
            cesm_vin2p(i, j)

if __name__ == '__main__':
    main()