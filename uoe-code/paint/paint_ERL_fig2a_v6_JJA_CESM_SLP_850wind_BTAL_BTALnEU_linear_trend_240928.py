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

v4 modified:
change to linear trend

v5 modified:
change to huaibei server
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
#import cmasher as cmr
from scipy.ndimage import gaussian_filter

sys.path.append('/home/sun/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# =================== File Information ==========================

# ------------------- PSL data ----------------------------------

file_path = '/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/'

psl_btal    = xr.open_dataset(file_path + 'PSL_BTAL_ensemble_mean_JJA_231020.nc')
psl_btalneu = xr.open_dataset(file_path + 'PSL_BTALnEU_ensemble_mean_JJA_231020.nc')
#print(psl_btal)

# ------------------- Wind data ---------------------------------

sel_level   = 850

u_btal      = xr.open_dataset(file_path + 'U_BTAL_ensemble_mean_JJA_231019.nc').sel(lev=sel_level)
v_btal      = xr.open_dataset(file_path + 'V_BTAL_ensemble_mean_JJA_231019.nc').sel(lev=sel_level)
u_btalneu   = xr.open_dataset(file_path + 'U_BTALnEU_ensemble_mean_JJA_231019.nc').sel(lev=sel_level)
v_btalneu   = xr.open_dataset(file_path + 'V_BTALnEU_ensemble_mean_JJA_231019.nc').sel(lev=sel_level)

# ------------------- Lat/Lon -----------------------------------

lat         = u_btal.lat.data

lon         = u_btal.lon.data

# =================== End of File Location ======================

# =================== Read data and calculate period difference ======================
period   = slice(1901, 1955)

# Claim the function for calculating linear trend for the input
def calculate_linear_trend(start, end, input_array, varname):
    from scipy.stats import linregress

    time_dim, lat_dim, lon_dim = input_array.sel(time=slice(start, end))[varname].shape

    trend_data = np.zeros((lat_dim, lon_dim))
    p_data     = np.zeros((lat_dim, lon_dim))

    input_data = input_array.sel(time=slice(start, end))[varname].data
    #print(input_data.shape)

    for i in range(lat_dim):
        for j in range(lon_dim):
            #print(linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j]))
            slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j])
            trend_data[i, j] = slope
            p_data[i, j]    = p_value

    return trend_data, p_value

p1 = 1901 ; p2 = 1955
u_con, u_p_con = calculate_linear_trend(p1, p2, u_btal, 'U_JJA')
v_con, v_p_con = calculate_linear_trend(p1, p2, v_btal, 'V_JJA')
u_neu, u_p_neu = calculate_linear_trend(p1, p2, u_btalneu, 'U_JJA')
v_neu, v_p_neu = calculate_linear_trend(p1, p2, v_btalneu, 'V_JJA')
psl_con, psl_p_con = calculate_linear_trend(p1, p2, psl_btal,    'PSL_JJA')
psl_neu, psl_p_neu = calculate_linear_trend(p1, p2, psl_btalneu, 'PSL_JJA')

print(np.nanmax(psl_con))

# ===================== END for function cal_period_difference ============================

# ===================== Painting function =======================

def plot_diff_slp_wind(diff_slp, diff_u, diff_v, left_title, right_title, out_path, pic_name, level):
    '''This function plot the difference in precipitation'''

    # ------------ colormap ----------------------------------
#    cmap = cmr.prinsenvlag

    # ------------ level -------------------------------------

    levels = level

    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 12), subplot_kw={'projection': proj})

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  60,125,5,35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(10,60,6,dtype=int),nx=1,ny=1,labelsize=20)

    # Shading for SLP difference
    im  =  ax.contourf(lon, lat, diff_slp, levels=levels, cmap='coolwarm', alpha=1, extend='both')
    
    # Vectors for Wind difference
    q  =  ax.quiver(lon, lat, diff_u, diff_v, 
                        regrid_shape=12.5, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
                        scale_units='xy', scale=1.6,        # scale是参考矢量，所以取得越大画出来的箭头就越短
                        units='xy', width=0.275,              # width控制粗细
                        transform=proj,
                        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)
    
    add_vector_legend(ax=ax, q=q, speed=5)

#    # Stippling picture
    #sp  =  ax.contourf(lon, lat, p_value, levels=[0., 0.1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='110m', lw=1.5)

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=25)
    ax.set_title(right_title, loc='right', fontsize=25)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=15)

    plt.savefig(out_path + pic_name)


def main():

    # --------------------- Part of painting ----------------------------------------------------------
    #data_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/ERL_fig2a_CESM_PSL_850UV_BTAL_BTALnEU.nc")

    out_path  = "/home/sun/paint/ERL/"
    level1    =  np.array([-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70,])
    level2    =  np.array([-28, -24, -20, -16, -12, -8,-4,0, 4, 8, 12, 16, 20, 24, 28], dtype=int)
    level2    =  np.linspace(-5, 5, 11)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_diff"],    diff_u=data_file["u_btal_diff"], diff_v=data_file["v_btal_diff"] , left_title='BTAL', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL.pdf", p=data_file['psl_btal_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btalneu_diff"], diff_u=data_file["u_btalneu_diff"], diff_v=data_file["v_btalneu_diff"] , left_title='BTALnEU', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTALnEU.pdf", p=data_file['psl_btalneu_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_btalneu_diff"],    diff_u=data_file["u_btal_btalneu_diff"], diff_v=data_file["v_btal_btalneu_diff"] , left_title='(a)', right_title='BTAL - BTALnEU', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL_BTALnEU.pdf", p=data_file['psl_btal_btalneu_diffp'], level=level2)
    plot_diff_slp_wind(diff_slp=1e1*gaussian_filter((psl_con - psl_neu), sigma=0.1), diff_u=(u_con - u_neu)*1e1, diff_v=(v_con - v_neu)*1e1,  left_title='(a)', right_title='CESM_ALL - CESM_noEU', out_path=out_path, pic_name="ERL_fig2a_v5_CESM_prect_diff_JJA_linear_trend_1901to1955.pdf", level=level2)


if __name__ == '__main__':
    main()
