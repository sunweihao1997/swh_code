'''
2025-4-9

Version Information:
https://www.notion.so/Points-With-Meeting-with-Massimot-1c1d5b19b11d8024bc2bff0b15f0f374?pvs=4
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
from matplotlib.ticker import FormatStrFormatter

sys.path.append('/home/sun/swh_code/uoe-code/module/')
from module_sun import set_cartopy_tick
from module_sun import check_path, add_vector_legend

# =================== File Information ==========================

# ------------------- PSL data ----------------------------------

#file_path = '/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/'
file_path = '/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/'

psl_btal    = xr.open_dataset(file_path + 'BTAL_SLP_jja_mean_241211.nc')
psl_btalneu = xr.open_dataset(file_path + 'noEU_SLP_jja_mean_241211.nc')
sf          = xr.open_dataset(file_path + 'Aerosol_Research_CESM_BTAL_BTALnEU_850hPa_streamfunction_velocity_potential.nc')
#print(psl_btal)

# ------------------- Wind data ---------------------------------
file_path = "/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/"

sel_level   = 850

u_btal      = xr.open_dataset(file_path + 'CESM_BTAL_JJA_U_ensemble.nc').sel(lev=sel_level)
v_btal      = xr.open_dataset(file_path + 'CESM_BTAL_JJA_V_ensemble.nc').sel(lev=sel_level)
u_btalneu   = xr.open_dataset(file_path + 'CESM_BTALnEU_JJA_U_ensemble.nc').sel(lev=sel_level)
v_btalneu   = xr.open_dataset(file_path + 'CESM_BTALnEU_JJA_V_ensemble.nc').sel(lev=sel_level)

ref_850 = xr.open_dataset(file_path + 'CESM_BTAL_JJA_U_ensemble.nc').sel(lev=850)

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

    #return trend_data, p_value
    return trend_data

p1 = 1901 ; p2 = 1955
u_con = (calculate_linear_trend(p1, p2, u_btal, 'JJA_U_1') + calculate_linear_trend(p1, p2, u_btal, 'JJA_U_2') + calculate_linear_trend(p1, p2, u_btal, 'JJA_U_3') + calculate_linear_trend(p1, p2, u_btal, 'JJA_U_4') + calculate_linear_trend(p1, p2, u_btal, 'JJA_U_5') + calculate_linear_trend(p1, p2, u_btal, 'JJA_U_6') + calculate_linear_trend(p1, p2, u_btal, 'JJA_U_7') + calculate_linear_trend(p1, p2, u_btal, 'JJA_U_8'))/8
v_con = (calculate_linear_trend(p1, p2, v_btal, 'JJA_V_1') + calculate_linear_trend(p1, p2, v_btal, 'JJA_V_2') + calculate_linear_trend(p1, p2, v_btal, 'JJA_V_3') + calculate_linear_trend(p1, p2, v_btal, 'JJA_V_4') + calculate_linear_trend(p1, p2, v_btal, 'JJA_V_5') + calculate_linear_trend(p1, p2, v_btal, 'JJA_V_6') + calculate_linear_trend(p1, p2, v_btal, 'JJA_V_7') + calculate_linear_trend(p1, p2, v_btal, 'JJA_V_8'))/8
u_neu = (calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_1') +calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_2') + calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_3') + calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_4') + calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_5') + calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_6') + calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_7') + calculate_linear_trend(p1, p2, u_btalneu, 'JJA_U_8'))/8
v_neu = (calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_1') +calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_2') + calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_3') + calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_4') + calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_5') + calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_6') + calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_7') + calculate_linear_trend(p1, p2, v_btalneu, 'JJA_V_8'))/8
psl_con= calculate_linear_trend(p1, p2, sf,    'btal_sf')
psl_neu= calculate_linear_trend(p1, p2, sf, 'btalneu_sf')

psl_diff = psl_con - psl_neu
#psl_diff[np.isnan(ref_850['JJA_U_2'].data[5])] = np.nan
psl_diff -= np.average(psl_diff, axis=1, keepdims=True)

#print(np.nanmean(psl_con))
#sys.exit("Succeed")

# ===================== END for function cal_period_difference ============================

# ===================== Painting function =======================

def plot_diff_slp_wind(diff_slp, diff_u, diff_v,
                       left_title, right_title,
                       out_path, pic_name, level):
    """
    只画 SLP 差值的等值线填色图，不画矢量。
    diff_u, diff_v 参数保留只是为了兼容旧调用，目前函数内部不使用。

    Parameters
    ----------
    diff_slp : 2D array (lat, lon)
        SLP 差值场（或其他标量场）
    diff_u, diff_v : 2D array
        兼容旧接口，当前未使用
    left_title, right_title : str
        左上 / 右上角标题
    out_path : str
        输出路径
    pic_name : str
        文件名
    level : 1D array
        等值线等级数组
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib import cm
    from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
    from cartopy.util import add_cyclic_point
    import os

    # ================== 1. 自定义 colormap ==================
    viridis   = cm.get_cmap('coolwarm')          # 以 coolwarm 为基础
    newcolors = viridis(np.linspace(0., 1, 256))
    midpoint  = 128                              # 白色所在索引，可调整
    newcolors[midpoint] = [1, 1, 1, 1]           # RGBA 白色
    newcmp    = LinearSegmentedColormap.from_list(
        "custom_coolwarm", newcolors
    )

    # ================== 2. 计算纬向偏差（可按需改掉） ==================
    sf_d = diff_slp.copy()
    for i in range(len(lat)):            # 假设 diff_slp 维度为 (lat, lon)
        sf_d[i] = sf_d[i] - np.nanmean(diff_slp[i])

    # 加环点，保证经向连续
    cyclic_sfd_vint, cyclic_lon = add_cyclic_point(sf_d, coord=lon)

    # ================== 3. 投影设置 ==================
    proj_data = ccrs.PlateCarree()                    # 数据所在坐标系：标准经纬度
    proj_map  = ccrs.PlateCarree(central_longitude=120)  # 画布投影：中心经线 120E

    fig, ax = plt.subplots(
        subplot_kw={'projection': proj_map},
        figsize=(10, 6)
    )

    # --- 范围 & 刻度 ---
    lonmin, lonmax, latmin, latmax = -5, 240, 10, 80
    extent = [lonmin, lonmax, latmin, latmax]
    ax.set_extent(extent, crs=proj_data)

    # 你的自定义刻度函数
    set_cartopy_tick(
        ax=ax,
        extent=extent,
        xticks=np.linspace(0, 210, 8, dtype=int),
        yticks=np.linspace(10, 80, 8, dtype=int),
        nx=1, ny=1,
        labelsize=6
    )

    # ================== 4. 等值线填色图 ==================
    # 使用传进来的 level 数组来做等级 & 归一化
    norm = BoundaryNorm(level, ncolors=256, clip=True)

    im = ax.contourf(
        cyclic_lon, lat, cyclic_sfd_vint,
        levels=level,             # ✅ 使用你指定的 level 数组，而不是 21
        cmap=newcmp,
        extend='both',
        norm=norm,
        transform=proj_data       # ✅ 关键：告诉 cartopy 数据是标准经纬度
    )

    # ================== 5. 海岸线 & 国界 ==================
    ax.coastlines(resolution='110m', lw=1.0)
    ax.add_feature(cfeature.BORDERS, linewidth=0.8)

    # ================== 6. 标题 ==================
    ax.set_title(left_title,  loc='left',  fontsize=14)
    ax.set_title(right_title, loc='right', fontsize=14)

    # ================== 7. 色标 ==================
    fig.subplots_adjust(bottom=0.18, top=0.92, left=0.07, right=0.97)

    cbar_ax = fig.add_axes([0.1, 0.08, 0.8, 0.03])
    cb = fig.colorbar(
    im,
    cax=cbar_ax,
    orientation='horizontal',
    ticks=level          # ✅ 每个 level 都画一个 tick
    )

# 如果想控制格式（比如保留一位小数）：
    cb.ax.set_xticklabels([f"{lv:.1f}" for lv in level])

    cb.ax.tick_params(labelsize=8)
    # 根据需要加单位
    # cb.set_label('SLP difference (units)', fontsize=10)

    # ================== 8. 保存图片 ==================
    if out_path is not None and pic_name is not None:
        os.makedirs(out_path, exist_ok=True)
        save_name = os.path.join(out_path, pic_name)
        plt.savefig(save_name, dpi=300, bbox_inches='tight')

    return fig, ax




def main():

    # --------------------- Part of painting ----------------------------------------------------------
    #data_file = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/ERL_fig2a_CESM_PSL_850UV_BTAL_BTALnEU.nc")

    out_path  = "/home/sun/paint/ERL/"
    level1    =  np.array([-70, -60, -50, -40, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70,])
    level2    =  np.array([-28, -24, -20, -16, -12, -8,-4,0, 4, 8, 12, 16, 20, 24, 28], dtype=int)
    level2    =  np.array([-6, -3, -1, -0.5, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 3, 6])
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_diff"],    diff_u=data_file["u_btal_diff"], diff_v=data_file["v_btal_diff"] , left_title='BTAL', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL.pdf", p=data_file['psl_btal_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btalneu_diff"], diff_u=data_file["u_btalneu_diff"], diff_v=data_file["v_btalneu_diff"] , left_title='BTALnEU', right_title='JJA', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTALnEU.pdf", p=data_file['psl_btalneu_diffp'], level=level1)
#    plot_diff_slp_wind(diff_slp=data_file["psl_btal_btalneu_diff"],    diff_u=data_file["u_btal_btalneu_diff"], diff_v=data_file["v_btal_btalneu_diff"] , left_title='(a)', right_title='BTAL - BTALnEU', out_path=out_path, pic_name="Aerosol_research_ERL_2a_BTAL_BTALnEU.pdf", p=data_file['psl_btal_btalneu_diffp'], level=level2)
#    level2    =  np.array([-5, -4.5, -4, -3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
    plot_diff_slp_wind(diff_slp=55*gaussian_filter((psl_diff/1e6), sigma=1), diff_u=(u_con - u_neu)*55, diff_v=(v_con - v_neu)*55,  left_title='(a)', right_title='CESM_ALL - CESM_noEU', out_path=out_path, pic_name="ERL_fig_response_CESM_st850_850wind_diff_JJA_linear_trend_1901to1955.pdf", level=level2)
    print('Finished')

if __name__ == '__main__':
    main()
