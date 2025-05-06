'''
2023-12-28
This script serves for the first picture of Fig2, including SLP and wind at 850 hPa

This script will plot three pictures:
period difference in BTAL
period difference in BTALnEU
influence of EU aerosol in the above difference
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
from scipy import stats
import cmasher as cmr

sys.path.append('/exports/csce/datastore/geos/users/s2618078/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# =================== File Information ==========================

# ------------------- PSL data ----------------------------------

file_path = '/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/'

psl_btal    = xr.open_dataset(file_path + 'PSL_BTAL_ensemble_mean_JJAS_231020.nc')
psl_btalneu = xr.open_dataset(file_path + 'PSL_BTALnEU_ensemble_mean_JJAS_231020.nc')

# ------------------- Wind data ---------------------------------

sel_level   = 850

u_btal      = xr.open_dataset(file_path + 'U_BTAL_ensemble_mean_JJAS_231019.nc').sel(lev=sel_level)
v_btal      = xr.open_dataset(file_path + 'V_BTAL_ensemble_mean_JJAS_231019.nc').sel(lev=sel_level)
u_btalneu   = xr.open_dataset(file_path + 'U_BTALnEU_ensemble_mean_JJAS_231019.nc').sel(lev=sel_level)
v_btalneu   = xr.open_dataset(file_path + 'V_BTALnEU_ensemble_mean_JJAS_231019.nc').sel(lev=sel_level)

# ------------------- Lat/Lon -----------------------------------

lat         = u_btal.lat.data
lon         = u_btal.lon.data

# =================== End of File Location ======================

# =================== Read data and calculate period difference ======================

def cal_period_difference_2d(data1, periodA, data2, periodB, varname,):
    '''
        This function calculate period difference

        periodA and periodB should be the format of slice
        under the normal circumstance data1 and data2 should be the same 
    '''
    lat = data1.lat.data
    lon = data1.lon.data
    time= data1.time.data

    # 1. calculate period difference
    data_pA = data1.sel(time=periodA) ; data_pB = data2.sel(time=periodB)

    data_period_diff = -1 * (np.average(data_pA[varname], axis=0) - np.average(data_pB[varname], axis=0))

    # 2. calculate student-t test
    p_value = np.ones((len(lat), len(lon)))

    for yy in range(len(lat)):
        for xx in range(len(lon)):
            anomaly_periodA = data_pA[varname].data[:, yy, xx] - np.average(data1[varname].data[:, yy, xx])
            anomaly_periodB = data_pB[varname].data[:, yy, xx] - np.average(data2[varname].data[:, yy, xx])

            a,b  = stats.ttest_ind(anomaly_periodA, anomaly_periodB, equal_var=False)
            p_value[yy, xx] = b

    print('{} complete'.format(varname))
    return data_period_diff, p_value

def cal_period_difference_3d(data1, periodA, data2, periodB, varname,):
    '''
        This function calculate period difference

        periodA and periodB should be the format of slice
        under the normal circumstance data1 and data2 should be the same 
    '''
    lat = data1.lat.data
    lon = data1.lon.data
    lev = data1.lev.data
    time= data1.time.data

    # 1. calculate period difference
    data_pA = data1.sel(time=periodA) ; data_pB = data2.sel(time=periodB)

    data_period_diff = -1 * (np.average(data_pA[varname], axis=0) - np.average(data_pB[varname], axis=0))

    # 2. calculate student-t test
    p_value = np.ones((len(lev), len(lat), len(lon)))

    for zz in range(len(lev)):
        for yy in range(len(lat)):
            for xx in range(len(lon)):
                anomaly_periodA = data_pA[varname].data[:, zz, yy, xx] - np.average(data1[varname].data[:, zz, yy, xx])
                anomaly_periodB = data_pB[varname].data[:, zz, yy, xx] - np.average(data2[varname].data[:, zz, yy, xx])

                a,b  = stats.ttest_ind(anomaly_periodA, anomaly_periodB, equal_var=False)
                p_value[zz, yy, xx] = b

    print('{} complete'.format(varname))
    return data_period_diff, p_value

# ===================== END for function cal_period_difference ============================

# ===================== Painting function =======================

def plot_diff_slp_wind(diff_slp, diff_u, diff_v, left_title, right_title, out_path, pic_name, p, level):
    '''This function plot the difference in precipitation'''

    # ------------ colormap ----------------------------------
    cmap = cmr.fusion

    # ------------ level -------------------------------------

    levels = level

    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 10), subplot_kw={'projection': proj})

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  55, 105, 0, 35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60, 100, 5,dtype=int), yticks=np.linspace(0, 30, 4, dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for SLP difference
    im  =  ax.contourf(lon, lat, diff_slp, levels=levels, cmap=cmap, alpha=1, extend='both')
    
    # Vectors for Wind difference
    q  =  ax.quiver(lon, lat, diff_u, diff_v, 
                        regrid_shape=20, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                        scale_units='xy', scale=0.05,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                        units='xy', width=0.25,              # width控制粗细
                        transform=proj,
                        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)
    
    add_vector_legend(ax=ax, q=q, speed=0.1)

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=17.5)
    ax.set_title(right_title, loc='right', fontsize=17.5)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=15, rotation=45)

    plt.savefig(out_path + pic_name)


def main():
    # data preparation

    periodA = slice(1905, 1920)
    periodB = slice(1945, 1960)

    # -------------------- Part of calculation which has completed ---------------------------
#    psl_btal_diff,    psl_btal_diffp       = cal_period_difference_2d(psl_btal,    periodA, psl_btal,    periodB, 'PSL_JJAS')
#    psl_btalneu_diff, psl_btalneu_diffp    = cal_period_difference_2d(psl_btalneu, periodA, psl_btalneu, periodB, 'PSL_JJAS')
#    psl_btal_btalneu_diff, psl_btal_btalneu_diffp = cal_period_difference_2d(psl_btal, periodB, psl_btalneu, periodB, 'PSL_JJAS')
#
#    u_btal_diff,      u_btal_diffp         = cal_period_difference_2d(u_btal,      periodA, u_btal,    periodB, 'U_JJAS')
#    u_btalneu_diff,   u_btalneu_diffp      = cal_period_difference_2d(u_btalneu,   periodA, u_btalneu, periodB, 'U_JJAS')
#    u_btal_btalneu_diff, u_btal_btalneu_diffp = cal_period_difference_2d(u_btal,   periodB, u_btalneu, periodB, 'U_JJAS')
#
#    v_btal_diff,      v_btal_diffp         = cal_period_difference_2d(v_btal,      periodA, v_btal,    periodB, 'V_JJAS')
#    v_btalneu_diff,   v_btalneu_diffp      = cal_period_difference_2d(v_btalneu,   periodA, v_btalneu, periodB, 'V_JJAS')
#    v_btal_btalneu_diff, v_btal_btalneu_diffp = cal_period_difference_2d(v_btal,   periodB, v_btalneu, periodB, 'V_JJAS')
#
#    ncfile  =  xr.Dataset(
#        {
#            "psl_btal_diff":     (["lat", "lon"], psl_btal_diff),
#            "psl_btal_diffp":    (["lat", "lon"], psl_btal_diffp),
#            "psl_btalneu_diff":  (["lat", "lon"], psl_btalneu_diff),
#            "psl_btalneu_diffp": (["lat", "lon"], psl_btalneu_diffp),
#            "u_btal_diff":       (["lat", "lon"], u_btal_diff),
#            "u_btal_diffp":      (["lat", "lon"], u_btal_diffp),
#            "u_btalneu_diff":    (["lat", "lon"], u_btalneu_diff),
#            "u_btalneu_diffp":   (["lat", "lon"], u_btalneu_diffp),
#            "v_btal_diff":       (["lat", "lon"], v_btal_diff),
#            "v_btal_diffp":      (["lat", "lon"], v_btal_diffp),
#            "v_btalneu_diff":    (["lat", "lon"], v_btalneu_diff),
#            "v_btalneu_diffp":   (["lat", "lon"], v_btalneu_diffp),
#            "v_btal_btalneu_diff":    (["lat", "lon"], v_btal_btalneu_diff),
#            "v_btal_btalneu_diffp":   (["lat", "lon"], v_btal_btalneu_diffp),
#            "psl_btal_btalneu_diff":  (["lat", "lon"], psl_btal_btalneu_diff * -1),
#            "psl_btal_btalneu_diffp": (["lat", "lon"], psl_btal_btalneu_diffp),
#            "u_btal_btalneu_diff":    (["lat", "lon"], u_btal_btalneu_diff * -1),
#            "u_btal_btalneu_diffp":   (["lat", "lon"], u_btal_btalneu_diffp * -1),
#
#        },
#        coords={
#            "lat":  (["lat"],  psl_btal['lat'].data),
#            "lon":  (["lon"],  psl_btal['lon'].data),
#        },
#        )
#
#    ncfile.attrs['description']  =  'Created on 2023-12-28. This file saves the PSL and UV (850) JJAS period difference and ttest p-value both for the BTAL and BTALnEU'
#
#    ncfile.to_netcdf("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/ERL_fig2a_CESM_PSL_850UV_BTAL_BTALnEU.nc", format='NETCDF4')
    # -------------------------------------------------------------------------------------------------

    # --------------------- Part of painting ----------------------------------------------------------
    data_file = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/ERL_fig2a_CESM_PSL_850UV_BTAL_BTALnEU.nc")

    out_path  = "/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/"
    level1    =  np.array([-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70,])
    level2    =  np.array([-50, -30, -20, -17.5, -15, -12.5, -10, -7.5, -5, -2.5, -0.5, 0, 0.5, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 30, 50])
    plot_diff_slp_wind(diff_slp=data_file["psl_btal_diff"],    diff_u=data_file["u_btal_diff"], diff_v=data_file["v_btal_diff"] , left_title='BTAL', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL.pdf", p=data_file['psl_btal_diffp'], level=level1)
    plot_diff_slp_wind(diff_slp=data_file["psl_btalneu_diff"], diff_u=data_file["u_btalneu_diff"], diff_v=data_file["v_btalneu_diff"] , left_title='BTALnEU', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTALnEU.pdf", p=data_file['psl_btalneu_diffp'], level=level1)
    plot_diff_slp_wind(diff_slp=data_file["psl_btal_btalneu_diff"],    diff_u=data_file["u_btal_btalneu_diff"], diff_v=data_file["v_btal_btalneu_diff"] , left_title='(a)', right_title='BTAL - BTALnEU', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL_BTALnEU.pdf", p=data_file['psl_btal_btalneu_diffp'], level=level2)

if __name__ == '__main__':
    main()
