'''
2023-12-28
This script serves for the first picture of Fig2, including SLP and wind at 850 hPa

This script will plot three pictures:
period difference in BTAL
period difference in BTALnEU
influence of EU aerosol in the above difference

2023-1-3 modified:
After meeting with Massimo I realized that the circulation is not consistent with the moisture transportation
I need check the data 
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
#print(psl_btal)

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
period1   = slice(1900, 1920)
period2   = slice(1940, 1960)

# BTAL period difference
u_btal_p1 = u_btal.sel(time=period1) ; u_btal_p2 = u_btal.sel(time=period2)

u_btal_diff = np.average(u_btal_p2['U_JJAS'].data, axis=0) - np.average(u_btal_p1['U_JJAS'].data, axis=0)

v_btal_p1 = v_btal.sel(time=period1) ; v_btal_p2 = v_btal.sel(time=period2)

v_btal_diff = np.average(v_btal_p2['V_JJAS'].data, axis=0) - np.average(v_btal_p1['V_JJAS'].data, axis=0)

psl_btal_p1 = psl_btal.sel(time=period1) ; psl_btal_p2 = psl_btal.sel(time=period2)

psl_btal_diff = np.average(psl_btal_p2['PSL_JJAS'].data, axis=0) - np.average(psl_btal_p1['PSL_JJAS'], axis=0)

# BTALnEU period difference
u_btalneu_p1 = u_btalneu.sel(time=period1) ; u_btalneu_p2 = u_btalneu.sel(time=period2)

u_btalneu_diff = np.average(u_btalneu_p2['U_JJAS'].data, axis=0) - np.average(u_btalneu_p1['U_JJAS'].data, axis=0)

v_btalneu_p1 = v_btalneu.sel(time=period1) ; v_btalneu_p2 = v_btalneu.sel(time=period2)

v_btalneu_diff = np.average(v_btalneu_p2['V_JJAS'].data, axis=0) - np.average(v_btalneu_p1['V_JJAS'].data, axis=0)

psl_btalneu_p1 = psl_btalneu.sel(time=period1) ; psl_btalneu_p2 = psl_btalneu.sel(time=period2)

psl_btalneu_diff = np.average(psl_btalneu_p2['PSL_JJAS'].data, axis=0) - np.average(psl_btalneu_p1['PSL_JJAS'], axis=0)

# 2. calculate student-t test
p_value = np.ones((len(lat), len(lon)))

# The following is the ttest between BTAL and BTALnEU experiment
for yy in range(len(lat)):
    for xx in range(len(lon)):
        anomaly_periodB_BTAL    = psl_btal_p2['PSL_JJAS'].data[:, yy, xx]    - np.average(psl_btal['PSL_JJAS'].data[:, yy, xx])
        anomaly_periodB_BTALnEU = psl_btalneu_p2['PSL_JJAS'].data[:, yy, xx] - np.average(psl_btalneu['PSL_JJAS'].data[:, yy, xx])
        a,b  = stats.ttest_ind(anomaly_periodB_BTAL, anomaly_periodB_BTALnEU, equal_var=False)
        p_value[yy, xx] = b

print(np.min(p_value))
print(np.max(p_value))




# ===================== END for function cal_period_difference ============================

# ===================== Painting function =======================

def plot_diff_slp_wind(diff_slp, diff_u, diff_v, left_title, right_title, out_path, pic_name, level):
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
    fig, ax =  plt.subplots(figsize=(15, 12), subplot_kw={'projection': proj})

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  55, 105, 0, 30
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60, 100, 5,dtype=int), yticks=np.linspace(0, 30, 4, dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for SLP difference
    im  =  ax.contourf(lon, lat, diff_slp, levels=levels, cmap=cmap, alpha=1, extend='both')
    
    # Vectors for Wind difference
    q  =  ax.quiver(lon, lat, diff_u, diff_v, 
                        regrid_shape=17, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                        scale_units='xy', scale=0.04,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                        units='xy', width=0.25,              # width控制粗细
                        transform=proj,
                        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)
    
    add_vector_legend(ax=ax, q=q, speed=0.1)

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p_value, levels=[0., 0.1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=22.5)
    ax.set_title(right_title, loc='right', fontsize=22.5)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=15, rotation=45)

    plt.savefig(out_path + pic_name)


def main():

    # --------------------- Part of painting ----------------------------------------------------------
    #data_file = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/ERL_fig2a_CESM_PSL_850UV_BTAL_BTALnEU.nc")

    out_path  = "/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/ERL/"
    level1    =  np.array([-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70,])
    level2    =  np.array([-50, -30, -20, -15, -10, -5, -0.5, 0.5, 5, 10, 15, 20, 30, 50])
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_diff"],    diff_u=data_file["u_btal_diff"], diff_v=data_file["v_btal_diff"] , left_title='BTAL', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL.pdf", p=data_file['psl_btal_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btalneu_diff"], diff_u=data_file["u_btalneu_diff"], diff_v=data_file["v_btalneu_diff"] , left_title='BTALnEU', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTALnEU.pdf", p=data_file['psl_btalneu_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_btalneu_diff"],    diff_u=data_file["u_btal_btalneu_diff"], diff_v=data_file["v_btal_btalneu_diff"] , left_title='(a)', right_title='BTAL - BTALnEU', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL_BTALnEU.pdf", p=data_file['psl_btal_btalneu_diffp'], level=level2)
    plot_diff_slp_wind(diff_slp=(psl_btal_diff - psl_btalneu_diff), diff_u=(u_btal_diff - u_btalneu_diff), diff_v=(v_btal_diff - v_btalneu_diff),  left_title='(a)', right_title='CESM_ALL - CESM_noEU', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL_BTALnEU.pdf", level=level2)


if __name__ == '__main__':
    main()
