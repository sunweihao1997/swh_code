'''
2024-7-19
This script is to calculate the linear trend of TS over Europe area between BTAL and BTALnEU
'''
import xarray as xr
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy.interpolate import RegularGridInterpolator
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
#from global_land_mask import globe
from matplotlib.colors import BoundaryNorm

sys.path.append('/home/sun/swh_code/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend


#def is_land(lat, lon):
#    """
#    判断给定的经纬度坐标是否位于陆地上。
#
#    参数：
#    lat -- 纬度（范围：-90 到 90）
#    lon -- 经度（范围：-180 到 180）
#
#    返回值：
#    如果坐标位于陆地上，返回 True；否则返回 False。
#    """
#    if lon > 180:
#        lon-=360
#    return globe.is_land(lat, lon)

# =================== File Information ==========================

# ------------------- PSL data ----------------------------------

file_path = '/home/sun/data/download_data/data/analysis_data/'

psl_btal    = xr.open_dataset(file_path + 'cesm_allf_fixEU_evap_diff_jja.nc')
sst         = xr.open_dataset(file_path + 'cesm_allf_fixEU_sst_diff_jja.nc')
mask = np.isnan(sst['pa'])             # True 的地方是 sst 缺测
psl_btal['pa'] = psl_btal['pa'].where(~mask)
#print(psl_btal.pa.data.shape)
#psl_btalneu = xr.open_dataset(file_path + 'BTALnEU_SST_jja_mean.nc')

# interpolate Hadsst to model
#psl_btal.interp(lon=psl_btalneu.lon.data, lat=psl_btalneu.lat.data)
#exit("Success")

# ------------------- Lat/Lon -----------------------------------

lat         = psl_btal.lat.data
lon         = psl_btal.lon.data



# ===================== Painting function =======================

def plot_diff_slp_wind(diff_slp, left_title, right_title, out_path, pic_name, level, pvalue):
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
    fig, ax =  plt.subplots(figsize=(20, 7), subplot_kw={'projection': proj})

    # Tick settings
#    cyclic_data_vint, cyclic_lon = add_cyclic_point(diff_slp, coord=lon)
#    cyclic_data_p,    cyclic_lon = add_cyclic_point(pvalue, coord=lon)


    # --- Set range ---
    lonmin,lonmax,latmin,latmax  =  40,110,-20,20
    extent     =  [lonmin,lonmax,latmin,latmax]
#
#    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(40,110,8,dtype=int),yticks=np.linspace(-20,20,5,dtype=int),nx=1,ny=1,labelsize=20)

    # 1. 构造插值器：原始网格 (lat, lon) -> diff_slp
    f = RegularGridInterpolator((lat, lon), diff_slp)

    # 2. 生成更密的经纬网格，比如放大 3 倍
    lat_new = np.linspace(lat.min(), lat.max(), len(lat) * 3)
    lon_new = np.linspace(lon.min(), lon.max(), len(lon) * 3)
    lon2d, lat2d = np.meshgrid(lon_new, lat_new)

    # 3. 在新网格上插值
    points = np.stack([lat2d.ravel(), lon2d.ravel()], axis=-1)
    diff_slp_new = f(points).reshape(lat2d.shape)

    # 4. 画等值线（这里继续用你之前的正红负蓝方案）
    colors = []
    for lev in levels:
        if lev < 0:
            colors.append('b')
        elif lev > 0:
            colors.append('r')
        else:
            colors.append('k')

    im = ax.contour(
        lon2d, lat2d, diff_slp_new,
        levels=levels,
        colors=colors,
        alpha=1
    )

    # 每隔一个 level 标注一次
    ax.clabel(
        im,
        levels=im.levels[::2],
        inline=True,
        fontsize=10,
        fmt='%g'
    )

    
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
    cb  =  fig.colorbar(im, shrink=0.9, pad=0.1, orientation='horizontal', ticks=levels)
    cb.ax.tick_params(labelsize=15)

    plt.savefig(out_path + pic_name)


def main():

    # --------------------- Part of painting ----------------------------------------------------------
    #data_file = xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/ERL_fig2a_CESM_PSL_850UV_BTAL_BTALnEU.nc")

    out_path  = "/home/sun/paint/ERL/"
    level1    =  np.array([-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70,])
    level2    =  np.array([-28, -24, -20, -16, -12, -8,-4,0, 4, 8, 12, 16, 20, 24, 28], dtype=int)
    level2    =  np.array([-2, -1.5, -1, -0.5, -0.25, 0.25, 0.5, 1, 1.5, 2,], dtype=float) / 10
#    level2    =  np.linspace(-0.7, 0.7, 11)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_diff"],    diff_u=data_file["u_btal_diff"], diff_v=data_file["v_btal_diff"] , left_title='BTAL', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL.pdf", p=data_file['psl_btal_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btalneu_diff"], diff_u=data_file["u_btalneu_diff"], diff_v=data_file["v_btalneu_diff"] , left_title='BTALnEU', right_title='JJAS', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTALnEU.pdf", p=data_file['psl_btalneu_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_btalneu_diff"],    diff_u=data_file["u_btal_btalneu_diff"], diff_v=data_file["v_btal_btalneu_diff"] , left_title='(a)', right_title='BTAL - BTALnEU', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL_BTALnEU.pdf", p=data_file['psl_btal_btalneu_diffp'], level=level2)
    plot_diff_slp_wind(diff_slp=psl_btal['pa'].data,left_title='1901-1955 Linear Trend', right_title='CESM_ALL - CESM_noEU', out_path=out_path, pic_name="ERL_fig_response_JJA_EVAP_linear_trend.pdf", level=level2, pvalue=None)
#    plot_diff_slp_wind(diff_slp=1e1*gaussian_filter((psl_con), sigma=0.5),          left_title='1901-1955 Linear Trend', right_title='CESM_ALL', out_path=out_path,             pic_name="Aerosol_research_ERL_s5a_BTAL_BTALnEU_TS_linear_trend.pdf", level=level2, pvalue=None)
#    plot_diff_slp_wind(diff_slp=1e1*gaussian_filter((psl_neu), sigma=0.5),          left_title='1901-1955 Linear Trend', right_title='CESM_noEU', out_path=out_path,            pic_name="Aerosol_research_ERL_s5b_BTAL_BTALnEU_TS_linear_trend.pdf", level=level2, pvalue=None)


if __name__ == '__main__':
    main()
