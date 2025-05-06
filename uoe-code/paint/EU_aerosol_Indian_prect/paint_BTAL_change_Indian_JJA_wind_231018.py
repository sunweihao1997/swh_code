'''
2023-10-18
This script calculate and paint 3D wind changes between two given periods
'''
import xarray as xr
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
import sys
from matplotlib import projections
import cartopy.crs as ccrs
import cartopy
from cartopy.util import add_cyclic_point
import matplotlib.pyplot as plt
sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# select two period to calculate difference
class period:
    ''' This class infludes the two periods for compare '''
    periodA_1 = 50 ; periodA_2 = 70
    periodB_1 = 90 ; periodB_2 = 110

def archive_interpolated_model_output():
    '''
        Because I interpolated the model output data and save them in the same path,
        which is inconvient for CDO CAT, so this function aims to archive them in the different folder
    '''
    exp_name = ['BTAL', 'BTALnEU']
    var_name = ['OMEGA', 'RELHUM', 'T', 'U', 'V', 'Z3']

    all_files = os.listdir('/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/')

    # archive folder structure expname/variable name/ensemble number
    for ii in exp_name:
        os.system('mkdir -p /exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ii)
        for jj in var_name:
            os.system('mkdir -p /exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ii + '/' + jj)
            path1  =  '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ii + '/' + jj + '/'
            for ee in range(8): # ensemble members
                path2 = path1 + ii + '_' + str(ee + 1) + '/'
                os.system('mkdir -p ' + path2)
                for ff in all_files:
                    if '_'+ii+'_'+str(ee + 1) in ff and jj+'_B' in ff:
                        os.system('mv /exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun2/' + ff + ' ' + path2 + ff)
                        print('Successfully move the file ' + ff)

class calculate_class:
    out_path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/WIND/'
    member_num = 8
    def cdo_ensemble(path1, exp_name, var_name, member_num):
        '''This function cdo the emsemble member result'''
        import os

        out_path1 = calculate_class.out_path0 + exp_name + '/' + var_name + '/'
        check_path(out_path1, 1)
        for i in range(member_num):
            path0 = path1 + exp_name + '/' + var_name + '/' + exp_name + '_' + str(i + 1) + '/'
            
            os.system('cdo cat ' + path0 + '*nc ' + out_path1 + exp_name + '_' + var_name + '_' + '1850_150years_member_' + str(i + 1) + '.nc')

    ensemble_path0 = '/exports/csce/datastore/geos/users/s2618078/data/model_data/WIND/'
    def cal_ensemble(exp_name, var_name):
        '''This function calculate the ensemble mean among all the members'''
        ensemble_path1 = calculate_class.ensemble_path0 + exp_name + '/' + var_name + '/' # This path saves the ncfiles
        file_list = os.listdir(ensemble_path1)
        ref_file  = xr.open_dataset(ensemble_path1 + file_list[0])

        # === Claim the array for result ===
        var_3d = np.zeros((1883, 29, 96, 144))

        for i in range(calculate_class.member_num):
            f0 = xr.open_dataset(ensemble_path1 + exp_name + '_' + var_name + '_' + '1850_150years_member_' + str(i + 1) +'.nc')

            var_3d += f0[var_name].data/calculate_class.member_num

        print('Successfully calculated result for {} {}'.format(exp_name, var_name))

        # Write to nc file
        ncfile  =  xr.Dataset(
            {
                var_name: (["time", "lev", "lat", "lon"], var_3d),
            },
            coords={
                "time": (["time"], ref_file['time'].data),
                "lat":  (["lat"],  ref_file['lat'].data),
                "lon":  (["lon"],  ref_file['lon'].data),
                "lev":  (["lev"],  ref_file['lev'].data),
            },
            )

        ncfile[var_name].attrs = ref_file[var_name].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-19. This file is the ensemble mean among the 8 member in the {} emission experiments. The variable is {}.'.format(exp_name, var_name)
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/{}_{}_ensemble_mean_231019.nc".format(exp_name, var_name), format='NETCDF4')
        ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_wind_231018.py'

    def cal_jja_mean(file_path, file_name, var_name, exp_name,):
        '''This function calculate JJA mean for the data, which has been cdo cat and Ensemble_Mean'''
        file0 = xr.open_dataset(file_path + file_name)

        # === Claim 150 year array ===
        jja_mean = np.zeros((150, 29, 96, 144))

        for yyyy in range(150):
            jja_mean[yyyy] = np.average(file0[var_name].data[yyyy * 12 + 5 : yyyy * 12 + 8], axis=0)
        
        print('{} {} JJA mean calculation succeed!'.format(exp_name, var_name))

        # === Write to ncfile ===
        ncfile  =  xr.Dataset(
            {
                "{}_JJA".format(var_name): (["time", "lev", "lat", "lon"], jja_mean),
            },
            coords={
                "time": (["time"], np.linspace(1850, 1999, 150)),
                "lat":  (["lat"],  file0['lat'].data),
                "lon":  (["lon"],  file0['lon'].data),
                "lev":  (["lev"],  file0['lev'].data),
            },
            )

        ncfile["{}_JJA".format(var_name)].attrs = file0[var_name].attrs

        ncfile.attrs['description']  =  'Created on 2023-10-19. This file is JJA mean for the {}. Experiment is {}.'.format(var_name, exp_name)
        ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/{}_{}_ensemble_mean_JJA_231019.nc".format(var_name, exp_name), format='NETCDF4')
        ncfile.attrs['script']       =  'paint_BTAL_change_Indian_JJA_wind_231018.py'

    def cal_two_periods_difference(data, dim):
        ref_file    = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/model_data/WIND/BTAL/U/BTAL_U_1850_150years_member_1.nc')

        avg_periodA = np.average(data[period.periodA_1 : period.periodA_2], axis=0)
        avg_periodB = np.average(data[period.periodB_1 : period.periodB_2], axis=0)

        # Add coordination to the array
        if dim==3:
            ncfile  =  xr.Dataset(
                {
                    "jja_diff": (["lev", "lat", "lon"], avg_periodB - avg_periodA),
                },
                coords={
                    "lat":  (["lat"],  ref_file['lat'].data),
                    "lon":  (["lon"],  ref_file['lon'].data),
                    "lev":  (["lev"],  ref_file['lev'].data),
                },
                )
        else:
            ncfile  =  xr.Dataset(
                {
                    "jja_diff": (["lat", "lon"], avg_periodB - avg_periodA),
                },
                coords={
                    "lat":  (["lat"],  ref_file['lat'].data),
                    "lon":  (["lon"],  ref_file['lon'].data),
                },
                )
        
        return ncfile

class paint_class:
    '''This class save the settings and functions for painting'''
    # !!!!!!!!!! Settings !!!!!!!!!!!!
    # ======== Extent ==========
    #lonmin,lonmax,latmin,latmax  =  40,115,-10,40
    lonmin,lonmax,latmin,latmax  =  0,150,-20,75
    extent     =  [lonmin,lonmax,latmin,latmax]

    # ======== Set Figure projection ==========
    proj       =  ccrs.PlateCarree()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def plot_uv_diff_at_select_lev(diff_u, diff_v, diff_slp, lev, plot_name):
        '''This function is to plot difference among two periods'''
        proj       =  ccrs.PlateCarree()

        ax         =  plt.subplot(projection=proj)

        # Extract level data
        u          =  diff_u.sel(lev=lev)['jja_diff'].data ; v          =  diff_v.sel(lev=lev)['jja_diff'].data

        # Tick settings
        cyclic_data_u, cyclic_lon = add_cyclic_point(u, coord=diff_u['lon'].data)
        cyclic_data_v, cyclic_lon = add_cyclic_point(v, coord=diff_u['lon'].data)
        cyclic_data_slp, cyclic_lon = add_cyclic_point(diff_slp['jja_diff'].data, coord=diff_u['lon'].data)
        set_cartopy_tick(ax=ax,extent=paint_class.extent,xticks=np.linspace(-10,150,9,dtype=int),yticks=np.linspace(-20,60,5,dtype=int),nx=1,ny=1,labelsize=12)

        im  =  ax.contourf(cyclic_lon, diff_u['lat'].data, cyclic_data_slp, np.linspace(-50,50,11), cmap='coolwarm', alpha=1, extend='both')

        ax.coastlines(resolution='110m', lw=1.25)

        # Vector Map
        q  =  ax.quiver(cyclic_lon, diff_u['lat'].data, cyclic_data_u, cyclic_data_v, 
            regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.05,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.55,              # width控制粗细
            transform=proj,
            color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

        ax.set_title('1901-1921 to 1941-1961',fontsize=17.5)

        add_vector_legend(ax=ax, q=q, speed=0.25)

        plt.colorbar(im, orientation='horizontal')

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/wind/{}_period_diff_JJA_Indian_UV.pdf'.format(plot_name), dpi=500)

        ax.remove()

    def plot_uv_diff_at_select_lev_among_exps(diff_u1, diff_v1, diff_u2, diff_v2, diff_slp1, diff_slp2, lev,):
        '''This function is to plot difference among two periods'''
        proj       =  ccrs.PlateCarree()

        ax         =  plt.subplot(projection=proj)

        # Extract level data
        u1          =  diff_u1.sel(lev=lev)['jja_diff'].data ; v1         =  diff_v1.sel(lev=lev)['jja_diff'].data
        u2          =  diff_u2.sel(lev=lev)['jja_diff'].data ; v2         =  diff_v2.sel(lev=lev)['jja_diff'].data
        slp1        =  diff_slp1['jja_diff'].data            ; slp2       =  diff_slp2['jja_diff'].data 

        u           =  u1 - u2
        v           =  v1 - v2
        slp         =  slp1 - slp2

        # Tick settings
        cyclic_data_u, cyclic_lon = add_cyclic_point(u, coord=diff_u1['lon'].data)
        cyclic_data_v, cyclic_lon = add_cyclic_point(v, coord=diff_u1['lon'].data)
        cyclic_data_slp, cyclic_lon = add_cyclic_point(slp, coord=diff_u1['lon'].data)
        #set_cartopy_tick(ax=ax,extent=paint_class.extent,xticks=np.linspace(50,110,4,dtype=int),yticks=np.linspace(-10,40,6,dtype=int),nx=1,ny=1,labelsize=12)
        set_cartopy_tick(ax=ax,extent=paint_class.extent,xticks=np.linspace(0,150,6,dtype=int),yticks=np.linspace(-20,60,5,dtype=int),nx=1,ny=1,labelsize=12)

        im  =  ax.contourf(cyclic_lon, diff_u1['lat'].data, cyclic_data_slp, np.linspace(-35,35,15), cmap='coolwarm_r', alpha=1, extend='both')

        ax.coastlines(resolution='50m', lw=1.25)
        bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale='50m', facecolor='none', alpha=0.7)
        ax.add_feature(bodr, linestyle='--', edgecolor='k', alpha=1)

        # Vector Map
        q  =  ax.quiver(cyclic_lon, diff_u1['lat'].data, cyclic_data_u, cyclic_data_v, 
            regrid_shape=20, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
            scale_units='xy', scale=0.03,        # scale是参考矢量，所以取得越大画出来的箭头就越短
            units='xy', width=0.42,              # width控制粗细
            transform=proj,
            color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

        ax.set_title('1901-1921 to 1941-1961',fontsize=17.5)

        #add_vector_legend(ax=ax, q=q, speed=0.25)

        plt.colorbar(im, orientation='horizontal')

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/wind/BTAL_BTALnEU_period_diff_JJA_Indian_UV_large_scale.pdf', dpi=500)

        ax.remove()






def main():
    # === 1. Archive the intepolated modei output === !! Finished 2023-10-18-19:54!!
    #archive_interpolated_model_output()

    # === 2. Cdo cat the long-term files === !! Finished 2023-10-19-10:43!!
    src_path = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/'
    #calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTAL', var_name='U', member_num=8)
    #calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTAL', var_name='V', member_num=8)
    #calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTALnEU', var_name='U', member_num=8)
    #calculate_class.cdo_ensemble(path1 = src_path, exp_name='BTALnEU', var_name='V', member_num=8)

    # === 3. Calculate ensemble mean among the members output === !! Finished 2023-10-19-11:27!!
    #calculate_class.cal_ensemble('BTAL', 'U')
    #calculate_class.cal_ensemble('BTAL', 'V')
    #calculate_class.cal_ensemble('BTALnEU', 'U')
    #calculate_class.cal_ensemble('BTALnEU', 'V')

    # === 4. Calculate JJA mean === !!Finished 2023-10-18-09:56!! 
    analysis_data_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
#    calculate_class.cal_jja_mean(file_path=analysis_data_path, 
#                                file_name='BTAL_U_ensemble_mean_231019.nc', 
#                                var_name='U',
#                                exp_name='BTAL',
#                                )
#    calculate_class.cal_jja_mean(file_path=analysis_data_path, 
#                                file_name='BTAL_V_ensemble_mean_231019.nc', 
#                                var_name='V',
#                                exp_name='BTAL',
#                                )
#    calculate_class.cal_jja_mean(file_path=analysis_data_path, 
#                                file_name='BTALnEU_U_ensemble_mean_231019.nc', 
#                                var_name='U',
#                                exp_name='BTALnEU',
#                                )
#    calculate_class.cal_jja_mean(file_path=analysis_data_path, 
#                                file_name='BTALnEU_V_ensemble_mean_231019.nc', 
#                                var_name='V',
#                                exp_name='BTALnEU',
#                                )
    # === Step 4 !!Finished 2023-10-19:14:26!!===

    # === 5. Calculate difference between two given period ===
    # This step is quick, so I calculate every time instead of writing to nc file
    file_con_u = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/U_BTAL_ensemble_mean_JJA_231019.nc')
    con_diff_u = calculate_class.cal_two_periods_difference(file_con_u['U_JJA'].data, 3)

    file_con_v = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/V_BTAL_ensemble_mean_JJA_231019.nc')
    con_diff_v = calculate_class.cal_two_periods_difference(file_con_v['V_JJA'].data, 3)

    file_eu_u  = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/U_BTALnEU_ensemble_mean_JJA_231019.nc')
    eu_diff_u  = calculate_class.cal_two_periods_difference(file_eu_u['U_JJA'].data, 3)

    file_eu_v  = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/V_BTALnEU_ensemble_mean_JJA_231019.nc')
    eu_diff_v  = calculate_class.cal_two_periods_difference(file_eu_v['V_JJA'].data, 3)

    # === Calculate SLP difference ===
    file_con_slp = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/PSL_BTAL_ensemble_mean_JJA_231020.nc')
    con_diff_slp = calculate_class.cal_two_periods_difference(file_con_slp['PSL_JJA'].data, 2)

    file_eu_slp  = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/PSL_BTALnEU_ensemble_mean_JJA_231020.nc')
    eu_diff_slp  = calculate_class.cal_two_periods_difference(file_eu_slp['PSL_JJA'].data, 2)

    print('Successfully calculated period-difference in all variables')
    #print(eu_diff_u)
    
    # ================= 6. Painting section =========================================
    # === 6.1 Period difference among the control experiment and noEU experiment themsleves ===
    #paint_class.plot_uv_diff_at_select_lev(con_diff_u, con_diff_v, con_diff_slp, 850, plot_name='BTAL')
    #paint_class.plot_uv_diff_at_select_lev(eu_diff_u,  eu_diff_v,  eu_diff_slp,  850, plot_name='BTALnEU')

    paint_class.plot_uv_diff_at_select_lev_among_exps(con_diff_u,  con_diff_v, eu_diff_u, eu_diff_v, con_diff_slp, eu_diff_slp, 850,)
    #print(np.nanmin(con_diff_u['jja_diff']))



if __name__ == '__main__':
    main()