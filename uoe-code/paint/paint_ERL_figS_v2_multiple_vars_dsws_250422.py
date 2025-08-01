'''
2025-3-19
This script is to calculate the linear trend of TS over Europe area between BTAL and BTALnEU
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
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm


sys.path.append('/home/sun/swh_code/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# =================== File Information ==========================

# ------------------- PSL data ----------------------------------

file_path = '/home/sun/data/download_data/data/Supplement_Data_send_by_Massimo/'

file_var    = xr.open_dataset(file_path + 'cesm_allf_fixEU_dsws_trend_jja.nc')
#print(psl_btal)


# ------------------- Lat/Lon -----------------------------------

lat         = file_var.lat.data
lon         = file_var.lon.data

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

    return trend_data, p_data

def calculate_linear_trend_diff(start, end, input_array_btal, input_array_btalneu, varname):
    '''This function only for calculating the significance test'''
    from scipy.stats import linregress

    time_dim, lat_dim, lon_dim = input_array_btal.sel(time=slice(start, end))[varname].shape

    trend_data = np.zeros((lat_dim, lon_dim))
    p_data     = np.zeros((lat_dim, lon_dim))

    input_data1 = input_array_btal.sel(time=slice(start, end))[varname].data
    input_data2 = input_array_btalneu.sel(time=slice(start, end))[varname].data
    #print(input_data.shape)

    for i in range(lat_dim):
        for j in range(lon_dim):
            #print(linregress(np.linspace(1, time_dim, time_dim), input_data[:, i, j]))
            slope, intercept, r_value, p_value, std_err = linregress(np.linspace(1, time_dim, time_dim), input_data1[:, i, j] - input_data2[:, i, j])
            trend_data[i, j] = slope
            p_data[i, j]    = p_value

    return trend_data, p_data

#p1 = 1901 ; p2 = 1955
#psl_con, psl_p_con = calculate_linear_trend(p1, p2, psl_btal,    'TS_JJA')
#psl_neu, psl_p_neu = calculate_linear_trend(p1, p2, psl_btalneu, 'TS_JJA')
#psl_dif, psl_p_dif = calculate_linear_trend_diff(p1, 1960, psl_btal, psl_btalneu, 'TS_JJA')

# calculate the student-t test
def calculate_student_t_test(ncfile1, ncfile2, period1, period2, lat, lon, varname):
    # calculate t-test for the given period
    from scipy import stats

    ncfile_select1 = ncfile1.sel(time=slice(period1, period2))
    ncfile_select2 = ncfile2.sel(time=slice(period1, period2))
    p_value       = np.zeros((len(lat), len(lon)))
    for ii in range(len(lat)):
        for jj in range(len(lon)):
            a,b  = stats.ttest_ind(ncfile_select1[varname].data[:, ii, jj], ncfile_select2[varname].data[:, ii, jj], equal_var=False)
            p_value[ii, jj] = b

    ncfile = xr.DataArray(
                            data=p_value,
                            dims=["lat", "lon"],
                            coords=dict(
                                lon=(["lon"], lon),
                                lat=(["lat"], lat),
                            ),
                        )

    return ncfile

#p1 = 1940 ; p2 = 1960
#ncfile_p = psl_p_dif

#print(ncfile_p)
#sys.exit("Complete!")


#print(np.nanmax(psl_con))
#print(psl_p_con.shape)
#sys.exit()

# ===================== END for function cal_period_difference ============================

# ===================== Painting function =======================

def plot_diff_slp_wind(diff_slp, left_title, right_title, out_path, pic_name, level, pvalue):
    '''This function plot the difference in precipitation'''

    # ------------ colormap ----------------------------------
#    cmap = cmr.prinsenvlag

    # ------------ level -------------------------------------

    levels = level

    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap,LinearSegmentedColormap

    viridis = cm.get_cmap('coolwarm')
    newcolors = viridis(np.linspace(0., 1, 256))
    midpoint = 128  # 设置白色所在的索引（可以调整）
    newcolors[midpoint] = [1, 1, 1, 1]  # RGBA 表示白色
#    newcmp = ListedColormap(newcolors)
    newcmp = LinearSegmentedColormap.from_list("custom_coolwarm", newcolors)

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(15, 7), subplot_kw={'projection': proj})

    # Tick settings
    cyclic_data_vint, cyclic_lon = add_cyclic_point(diff_slp, coord=lon)
   # cyclic_data_p,    cyclic_lon = add_cyclic_point(pvalue, coord=lon)


    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  -15, 50, 40, 70
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(-15, 45, 5,dtype=int), yticks=np.linspace(20, 70, 6, dtype=int),nx=1,ny=1,labelsize=15)

    # Shading for SLP difference
    norm = BoundaryNorm(level, ncolors=256, clip=True)
    im   =  ax.contourf(cyclic_lon, lat, cyclic_data_vint, levels=levels, cmap='coolwarm', alpha=1, extend='both', norm=norm)
    #dot  =  ax.contourf(cyclic_lon, lat, cyclic_data_p, levels=[0., 0.15], colors='none', hatches=['.'])

    
    # Vectors for Wind difference
#    q  =  ax.quiver(lon, lat, diff_u, diff_v, 
#                        regrid_shape=15, angles='uv',        # regrid_shape这个参数越小，是两门就越稀疏
#                        scale_units='xy', scale=1.5,        # scale是参考矢量，所以取得越大画出来的箭头就越短
#                        units='xy', width=0.35,              # width控制粗细
#                        transform=proj,
#                        color='k', headlength = 5, headaxislength = 4, headwidth = 4, alpha=0.8)
    
    #add_vector_legend(ax=ax, q=q, speed=5)

#    # Stippling picture
    #sp  =  ax.contourf(lon, lat, p_value, levels=[0., 0.1], colors='none', hatches=['..'])

    # --- Coast Line ---
    ax.coastlines(resolution='110m', lw=1.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1)

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=15.5)
    ax.set_title(right_title, loc='right', fontsize=15.5)

    # ========= add colorbar =================
    cb  =  fig.colorbar(im, shrink=0.75, pad=0.1, orientation='horizontal', ticks=levels)
    cb.ax.tick_params(labelsize=12)

    plt.savefig(out_path + pic_name)


def main():

    # --------------------- Part of painting ----------------------------------------------------------
    #data_file = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/ERL_fig2a_CESM_PSL_850UV_BTAL_BTALnEU.nc")

    out_path  = "/home/sun/paint/ERL/"
    level1    =  np.array([-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70,])
    level2    =  np.array([-28, -24, -20, -16, -12, -8,-4,0, 4, 8, 12, 16, 20, 24, 28], dtype=int)
    #level2    =  np.array([-0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
    level2 = np.array([-10, -5, -4, -3, -2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 10])
    print(level2.shape)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_diff"],    diff_u=data_file["u_btal_diff"], diff_v=data_file["v_btal_diff"] , left_title='BTAL', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL.pdf", p=data_file['psl_btal_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btalneu_diff"], diff_u=data_file["u_btalneu_diff"], diff_v=data_file["v_btalneu_diff"] , left_title='BTALnEU', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTALnEU.pdf", p=data_file['psl_btalneu_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_btalneu_diff"],    diff_u=data_file["u_btal_btalneu_diff"], diff_v=data_file["v_btal_btalneu_diff"] , left_title='(a)', right_title='BTAL - BTALnEU', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL_BTALnEU.pdf", p=data_file['psl_btal_btalneu_diffp'], level=level2)
    plot_diff_slp_wind(diff_slp=gaussian_filter(file_var['pa'].data, sigma=1.), left_title='1901-1955 Linear Trend', right_title='dsws', out_path=out_path, pic_name="ERL_figSnew_JJA_BTAL_BTALnEU_dsws_linear_trend_v2.pdf", level=level2, pvalue=None)
#    plot_diff_slp_wind(diff_slp=1e1*gaussian_filter((psl_con), sigma=0.5),          left_title='1901-1955 Linear Trend', right_title='CESM_ALL', out_path=out_path,             pic_name="Aerosol_research_ERL_s5a_BTAL_BTALnEU_TS_linear_trend.pdf", level=level2, pvalue=None)
#    plot_diff_slp_wind(diff_slp=1e1*gaussian_filter((psl_neu), sigma=0.5),          left_title='1901-1955 Linear Trend', right_title='CESM_noEU', out_path=out_path,            pic_name="Aerosol_research_ERL_s5b_BTAL_BTALnEU_TS_linear_trend.pdf", level=level2, pvalue=None)


if __name__ == '__main__':
    main()
