'''
This edition: change to 1901 to 1955 linear trend on CESM BTAL experiment
'''
import xarray as xr
import numpy as np
from matplotlib import projections
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as mpatches
from scipy import stats
from scipy.ndimage import gaussian_filter
import cartopy.feature as cfeature


sys.path.append("/home/sun/uoe-code/module/")
from module_sun import *

lonmin,lonmax,latmin,latmax  =  60,125,5,35
extent     =  [lonmin,lonmax,latmin,latmax]

#levels = np.array([-1.8, -1.5, -1.2, -0.9, -0.6, -0.5, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.6, 0.9, 1.2, 1.5, 1.8])
#levels = np.linspace(-1., 1., 21)
#levels = np.array([-0.9, -0.6, -0.5, -0.4, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.9,])


# ===================== Calculation for JJA and JJAS precipitation period difference ===========================

data_path = "/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/"
data_file = "BTAL_precipitation_jja_mean_231005.nc"

data = xr.open_dataset(data_path + data_file)
lat  = data.lat.data
lon  = data.lon.data

start0 = 1901 ; end0 = 1955
years_x = np.linspace(start0, end0, end0 - start0 + 1)
data_01to55 = data.sel(time=slice(start0, end0))

jja_trend  = np.zeros((len(lat), len(lon)))
jjas_trend = np.zeros((len(lat), len(lon)))
jja_p  = np.zeros((len(lat), len(lon)))
jjas_p = np.zeros((len(lat), len(lon)))

for i in range(len(lat)):
    for j in range(len(lon)):
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(start0, end0, end0 - start0 + 1), data_01to55['PRECT_JJA'].data[:, i, j])
        jja_trend[i, j]  = slope
        jja_p[i, j]      = p_value

        A = np.vstack([years_x, np.ones_like(years_x)]).T
        slope, intercept = np.linalg.lstsq(A, data_01to55['PRECT_JJA'].data[:, i, j], rcond=None)[0] ; print('Ho ye')
        jja_trend[i, j]  = slope

#        slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(start0, end0, end0 - start0 + 1), data_01to55['PRECT_JJAS_B'].data[:, i, j])
#        jjas_trend[i, j] = slope
#        jjas_p[i, j]     = p_value



# Write trend into file
ncfile  =  xr.Dataset(
    {
        "JJA_trend":  (["lat", "lon"], jja_trend),
        "JJA_p":  (["lat", "lon"], jja_p),
    },
    coords={
        "lat":  (["lat"],  lat),
        "lon":  (["lon"],  lon),
    },
    )

ncfile["JJA_trend"].attrs['units']  = 'mm day^-1 year^-1'

ncfile.attrs['description'] = 'Created on 2024-12-5 by /home/sun/uoe-code/paint/paint_ERL_fig1c_v4_CESM_PRECT_BTAL_linear_trend_1901to1955_241205.py. Please note this is corrected version using the right JJA mean'
#
out_path = '/home/sun/data/download_data/data/analysis_data/'
ncfile.to_netcdf(out_path + 'Aerosol_Research_CESM_BTAL_PRECT_JJA_linear_trend_1901to1955_corrected.nc')


def plot_diff_rainfall(diff_data, left_title, right_title, out_path, pic_name, level, p):
    '''This function plot the difference in precipitation'''
    # ------------ 2. Paint the Pic --------------------------
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    # 2.2 Set the figure
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(20, 14), subplot_kw={'projection': proj})

    viridis = cm.get_cmap('coolwarm_r')
    newcolors = viridis(np.linspace(0., 1, 256))
    midpoint = 128  # 设置白色所在的索引（可以调整）
    newcolors[midpoint] = [1, 1, 1, 1]  # RGBA 表示白色
#    newcmp = ListedColormap(newcolors)
    newcmp = LinearSegmentedColormap.from_list("custom_coolwarm", newcolors)
    #newcmp.set_under('white')
    #newcmp.set_over('#145DA0')

#    # --- Set range ---
#    lonmin,lonmax,latmin,latmax  =  65,93,5,35
#    extent     =  [lonmin,lonmax,latmin,latmax]
#
#    # --- Tick setting ---
#    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(70,90,3,dtype=int),yticks=np.linspace(0,40,5,dtype=int),nx=1,ny=1,labelsize=15)

    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  40,130,0,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(0,50,6,dtype=int),nx=1,ny=1,labelsize=30)

    # Shading for precipitation trend
    im  =  ax.contourf(lon, lat, diff_data, levels=level, cmap=newcmp, alpha=1, extend='both')

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.15], colors='none', hatches=['.'])

    # --- Coast Line ---
    ax.coastlines(resolution='110m', lw=1.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1)

    # --- add patch for key area ---
    ax.add_patch(mpatches.Rectangle(xy=[72, 20], width=12, height=7.5,linestyle='--',
                                facecolor='none', edgecolor='grey', linewidth=3.5,
                                transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title(left_title, loc='left', fontsize=25)
    ax.set_title(right_title, loc='right', fontsize=25)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=20)

    cb.set_ticks(np.array([-0.6, -0.4,  -0.2, -0.05, 0.05, 0.2, 0.4, 0.6]))  # 自定义刻度位置
    cb.set_ticklabels(np.array([-0.6, -0.4,  -0.2, -0.05, 0.05, 0.2, 0.4, 0.6]))  # 自定义标签

    plt.savefig(out_path + pic_name)
    

def main():
    out_path = '/home/sun/paint/ERL/'

    lev0 = np.array([-0.12, -.1, -0.08, -0.06, -0.04, -0.02, 0, .02, .04, 0.06, 0.08, .1, .12,])
    lev0 = np.linspace(-.5, .5, 11)
    lev0 = np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    #plot_diff_rainfall(diff_data=prect_BTAL_JJA_DIFF, left_title='BTAL', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTAL_JJA_period_difference_1900_1960_231221.pdf")
    #plot_diff_rainfall(diff_data=prect_BTALnEU_JJA_DIFF,  left_title='BTALnEU', right_title='JJA',  out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJA_period_difference_1900_1960_231221.pdf")
    plot_diff_rainfall(diff_data=gaussian_filter(ncfile['JJA_trend'].data * 55 * 86400000, sigma=0.9), left_title='(c)', right_title='CESM_ALL (JJA)', out_path=out_path, pic_name="ERL_fig1c_v4_CESM_prect_BTAL_JJA_linear_trend_1901to1955.pdf", level=lev0, p=ncfile['JJA_p'].data)
#    plot_diff_rainfall(diff_data=prect_BTALnEU_JJAS_DIFF, left_title='(b)', right_title='CESM_noEU (JJAS)', out_path=out_path, pic_name="Aerosol_research_CESM_prect_BTALnEU_JJAS_period_difference_1900_1960_231221.pdf", p=p_value_BTALnEU)
#    plot_diff_rainfall(diff_data=(prect_BTAL_JJAS_DIFF - prect_BTALnEU_JJAS_DIFF), left_title='(d)', right_title='CESM_ALL - CESM_noEU (JJAS)', out_path=out_path, pic_name="ERL_fig1d_CESM_prect_BTAL_sub_BTALnEU_JJAS_period_difference_1900_1960_231227.pdf", p=p_value_BTAL_BTALnEU)


if __name__ == '__main__':
    main()