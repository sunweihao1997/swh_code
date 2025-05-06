'''
2023-10-25
This script is to plot the moisture flux and vertical moisture flux convergence

file information:
1. Moisture transportation (vector)
/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc
shape (1883,29,96,144)

2. Moisture flux vertical convergence (shading)
/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/BTAL_moisture_flux_integration_1850_150years.nc
shape (1883,96,144)

20231201 modified:
1. Previous calculation has error due to the start time of the result is 1850-02-01
2. The modification is to replace the wrong JJA to the JJAS with dt.month method
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
import concurrent.futures
import cmasher as cmr
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter

ref_file0 = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc")

# select two period to calculate difference
class period:
    ''' This class infludes the two periods for compare '''
    periodA_1 = 50 ; periodA_2 = 70
    periodB_1 = 90 ; periodB_2 = 110

class calculate_class:
    def cal_jjas_mean(file_path, file_name, var_name, exp_name, dim_num):
        '''This function calculate JJA mean for the data, which has been cdo cat and Ensemble_Mean'''
        ref_file0 = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc")
        file0     = xr.open_dataset(file_path + file_name)
        
        # === Claim 150 year array ===
        if dim_num == 3:
            jjas_mean = np.zeros((150, 29, 96, 144))
        else: 
            jjas_mean = np.zeros((150, 96, 144))

        if dim_num == 3:
            file1     = file0.sel(time=ref_file0.time.dt.month.isin([6, 7, 8, 9]))
            for yyyy in range(150):
                jjas_mean[yyyy] = np.average(file1[var_name].data[yyyy * 4 : yyyy * 4 + 4], axis=0)
        else:
            '''I need to save the raw data into the ncfile, that because the 2-D data was created from ncl and no coordination information was added'''
            ncfile0  =  xr.Dataset(
                {
                    var_name: (["time", "lat", "lon"], file0[var_name].data),
                },
                coords={
                    "time": (["time"], ref_file0.time.data),
                    "lat":  (["lat"],  ref_file0['lat'].data),
                    "lon":  (["lon"],  ref_file0['lon'].data),
                },
                )
            file1     = ncfile0.sel(time=ref_file0.time.dt.month.isin([6, 7, 8, 9]))
            for yyyy in range(150):
                jjas_mean[yyyy] = np.average(file1[var_name].data[yyyy * 4 : yyyy * 4 + 4], axis=0)
        
        print('{} {} JJAS mean calculation succeed!'.format(exp_name, var_name))

        # === Write to ncfile ===
        if dim_num == 3:
            ncfile  =  xr.Dataset(
                {
                    "{}_JJAS".format(var_name): (["time", "lev", "lat", "lon"], jjas_mean),
                },
                coords={
                    "time": (["time"], np.linspace(1850, 1999, 150)),
                    "lat":  (["lat"],  ref_file0['lat'].data),
                    "lon":  (["lon"],  ref_file0['lon'].data),
                    "lev":  (["lev"],  ref_file0['lev'].data),
                },
                )
        else:
            ncfile  =  xr.Dataset(
                {
                    "{}_JJAS".format(var_name): (["time", "lat", "lon"], jjas_mean),
                },
                coords={
                    "time": (["time"], np.linspace(1850, 1999, 150)),
                    "lat":  (["lat"],  ref_file0['lat'].data),
                    "lon":  (["lon"],  ref_file0['lon'].data),
                },
                )

        return ncfile

    def cal_two_periods_difference(data,):
        ref_file    = xr.open_dataset('/exports/csce/datastore/geos/users/s2618078/data/model_data/WIND/BTAL/U/BTAL_U_1850_150years_member_1.nc')

        avg_periodA = np.average(data[period.periodA_1 : period.periodA_2], axis=0)
        avg_periodB = np.average(data[period.periodB_1 : period.periodB_2], axis=0)

        return avg_periodB - avg_periodA


class paint_class:
    '''This class save the settings and functions for painting'''
    # !!!!!!!!!! Settings !!!!!!!!!!!!


    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    def plot_uv_diff_at_select_lev(f0, diff_u, diff_v, diff_vint, lev, plot_name, pvalue):
        '''This function is to plot difference among two periods'''
        # 2.2 Set the figure
        proj    =  ccrs.PlateCarree()
        fig, ax =  plt.subplots(figsize=(15, 12), subplot_kw={'projection': proj})

        # Tick settings
        cyclic_data_u, cyclic_lon = add_cyclic_point(diff_u, coord=f0['lon'].data)
        cyclic_data_v, cyclic_lon = add_cyclic_point(diff_v, coord=f0['lon'].data)
        cyclic_data_vint, cyclic_lon = add_cyclic_point(diff_vint, coord=f0['lon'].data)
        
        # --- Set range ---
        lonmin,lonmax,latmin,latmax  =  55, 125, 5, 40
        extent     =  [lonmin,lonmax,latmin,latmax]

        # --- Tick setting ---
        set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60, 130, 8,dtype=int), yticks=np.linspace(0, 40, 5, dtype=int),nx=1,ny=1,labelsize=25)

        level0 = np.array([-1.5, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.5])
        im  =  ax.contourf(cyclic_lon, f0['lat'].data, cyclic_data_vint * 1e4, level0, cmap='coolwarm', alpha=1, extend='both')

        # --- add patch for key area ---
        ax.add_patch(mpatches.Rectangle(xy=[72, 20], width=12, height=7.5,linestyle='--',
                                facecolor='none', edgecolor='grey', linewidth=3.5,
                                transform=ccrs.PlateCarree()))

        #plt.rcParams.update({'hatch.color': 'gray'})
        dot  =  ax.contourf(f0['lon'].data, f0['lat'].data, pvalue, levels=[0., 0.1], colors='none', hatches=['//'])

        ax.coastlines(resolution='50m', lw=1.5)

        #ax.plot([40,120],[0,0],'--', color='gray')

        # Vector Map
#        q  =  ax.quiver(cyclic_lon, f0['lat'].data, cyclic_data_u * 10, cyclic_data_v * 10, 
#            regrid_shape=12.7, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
#            scale_units='xy', scale=0.4,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#            units='xy', width=0.3,              # width控制粗细
#            transform=proj,
#            color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)

        ax.set_title('(b)', loc='left', fontsize=25)
        ax.set_title('CESM_ALL - CESM_noEU',loc='right', fontsize=25)

        # ========= add colorbar =================
        fig.subplots_adjust(top=0.8) 
        cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
        cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
        cb.ax.set_xticks(level0)
        cb.ax.tick_params(labelsize=15)

        plt.savefig('/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/{}_period_diff_JJA_Indian_UV.pdf'.format(plot_name), dpi=500)

        #ax.remove()

def calculate_student_t_test(ncfile, period1, period2, lat, lon):
    # calculate t-test for the given period
    from scipy import stats

    ncfile_select = ncfile.sel(time=slice(period1, period2))
    p_value       = np.zeros((len(lat), len(lon)))
    for ii in range(len(lat)):
        for jj in range(len(lon)):
            a,b  = stats.ttest_ind(ncfile_select['mt_vint_BTAL_JJAS'].data[:, ii, jj], ncfile_select['mt_vint_BTALnEU_JJAS'].data[:, ii, jj], equal_var=False)
            p_value[ii, jj] = b

    ncfile['p_value'] = xr.DataArray(
                            data=p_value,
                            dims=["lat", "lon"],
                            coords=dict(
                                lon=(["lon"], lon),
                                lat=(["lat"], lat),
                            ),
                        )

    # Save to the ncfile
    #ncfile.to_netcdf("/mnt/d/samssd/precipitation/processed/EUI_CESM_BTAL_fixEU_JJAS_precipitation_anomaly_150years_and_ttest_1940_to_1960.nc")
    return ncfile

def main():
    # ================= 1. Calculate JJA mean ========================
    path0 = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'
    path1 = '/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/'
#    with concurrent.futures.ProcessPoolExecutor() as executor:
#        a1 = executor.submit(calculate_class.cal_jjas_mean, file_path=path0, 
#                                file_name='BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc', 
#                                var_name='transport_x_BTAL',
#                                exp_name='BTAL',
#                                dim_num=3,
#                                )
#        a2 = executor.submit(calculate_class.cal_jjas_mean, file_path=path0, 
#                                file_name='BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc', 
#                                var_name='transport_y_BTAL',
#                                exp_name='BTAL',
#                                dim_num=3,
#                                )
#        a3 = executor.submit(calculate_class.cal_jjas_mean, file_path=path0, 
#                                file_name='BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc', 
#                                var_name='transport_x_BTALnEU',
#                                exp_name='BTALnEU',
#                                dim_num=3,
#                                )
#        a4 = executor.submit(calculate_class.cal_jjas_mean, file_path=path0, 
#                                file_name='BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc', 
#                                var_name='transport_y_BTALnEU',
#                                exp_name='BTALnEU',
#                                dim_num=3,
#                                )
#        a5 = executor.submit(calculate_class.cal_jjas_mean, file_path=path1, 
#                                file_name='BTAL_moisture_flux_integration_1850_150years.nc', 
#                                var_name='mt_vint_BTAL',
#                                exp_name='BTAL',
#                                dim_num=2,
#                                )
#        a6 = executor.submit(calculate_class.cal_jjas_mean, file_path=path1, 
#                                file_name='BTAL_moisture_flux_integration_1850_150years.nc', 
#                                var_name='mt_vint_BTALnEU',
#                                exp_name='BTALnEU',
#                                dim_num=2,
#                                )
#    
#    merge_array = xr.merge([a1.result(), a2.result(), a3.result(), a4.result(), a5.result(), a6.result()])
#    #print(merge_array)
##    print(a5.result())
#    # =================== 2. Calculate difference among two periods =================
#    # Directly write to ncfile
#    ncfile0 = calculate_student_t_test(ncfile=merge_array, period1=1945, period2=1960, lat=merge_array.lat.data, lon=merge_array.lon.data)
#    ncfile  =  xr.Dataset(
#    {
#        "transport_x_BTAL_diff": (["lev", "lat", "lon"], calculate_class.cal_two_periods_difference(merge_array['transport_x_BTAL_JJAS'].data)),
#        "transport_y_BTAL_diff": (["lev", "lat", "lon"], calculate_class.cal_two_periods_difference(merge_array['transport_y_BTAL_JJAS'].data)),
#        "transport_x_BTALnEU_diff": (["lev", "lat", "lon"], calculate_class.cal_two_periods_difference(merge_array['transport_x_BTALnEU_JJAS'].data)),
#        "transport_y_BTALnEU_diff": (["lev", "lat", "lon"], calculate_class.cal_two_periods_difference(merge_array['transport_y_BTALnEU_JJAS'].data)),
#        "mt_vint_BTAL_diff": (["lat", "lon"],    calculate_class.cal_two_periods_difference(merge_array['mt_vint_BTAL_JJAS'].data)),
#        "mt_vint_BTALnEU_diff": (["lat", "lon"], calculate_class.cal_two_periods_difference(merge_array['mt_vint_BTALnEU_JJAS'].data)),
#        "p_value": (["lat", "lon"], ncfile0['p_value'].data),
#    },
#    coords={
#        "lat":  (["lat"],  ref_file0['lat'].data),
#        "lon":  (["lon"],  ref_file0['lon'].data),
#        "lev":  (["lev"],  ref_file0['lev'].data),
#    },
#    )
##
#    ##print(ncfile)
#    print(" =========== Successfully calculate the period difference ===============")
#    # =================== 3. Calculate the ttest =====================================
#    ncfile.attrs['description']  =  "Created on 2023-10-25, This script save the array of the difference in long-term trend among the two experiments. 2023-12-1 update: Correct the wrong time selecting using dt.month method and change to the JJAS mean. 2023-12-3 update: Add ttest among the experiments for the period 1945-1960 fot the vertical integral of moisture transportation."
#    ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_difference_moisture_transportation_diff_1900_1960_231201_corrected_231203_ttest.nc")




    # =================== 4. Paint the picture =======================================
    # 3.1 Collect the information
    file0 = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_difference_moisture_transportation_diff_1900_1960_231201_corrected_231203_ttest.nc")
    #print(np.nanmax(file0["transport_y_BTAL_diff"].data))  unit：g m-1 s-1 hPa-1
    #print(np.nanmin(file0["transport_y_BTAL_diff"].data))
    #print(np.nanmax(file0["mt_vint_BTAL_diff"].data))      unit: g m-2 s-1
    #print(np.nanmin(file0["mt_vint_BTALnEU_diff"].data))

    # moisture transportation 10e-1
    # convergence             10e-4

    # Plot caution
    # 1. Select the level before passing the arguments
    lev0 = 850
    file0_lev = file0.sel(lev=850)
    paint_class.plot_uv_diff_at_select_lev(f0=file0, diff_u=(file0_lev["transport_x_BTAL_diff"].data - file0_lev["transport_x_BTALnEU_diff"].data), 
                                            diff_v=(file0_lev["transport_y_BTAL_diff"].data - file0_lev["transport_y_BTALnEU_diff"].data), 
                                            diff_vint=gaussian_filter((file0["mt_vint_BTAL_diff"].data - file0["mt_vint_BTALnEU_diff"].data), sigma=0.8), 
                                            lev=850, 
                                            plot_name='ERL_fig2b_v0_BTAL_BTALnEU_Water_transportation_period_difference',
                                            pvalue=file0_lev["p_value"].data)


if __name__ == '__main__':
    main()