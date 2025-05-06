'''
2024-7-7
This edition: change to 1901 to 1955 linear trend ; 2. interp into same grid

2024-9-14
This edition: 
move to huaibei server (path need to be changed)
other modifies
'''
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import sys
import matplotlib.patches as mpatches
from scipy import stats
from scipy.ndimage import gaussian_filter
from scipy.stats import theilslopes
import pymannkendall as mk
import cartopy.feature as cfeature
from matplotlib.colors import BoundaryNorm


module_path = '/home/sun/uoe-code/module/'
sys.path.append(module_path)
from module_sun import *

# ================ Calculation for 1901to1955 linear trend =======================
#ncfile1 = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/UDEL_JJA_JJAS_precip_1900_2017.nc")

ncfile  = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/UDEL_JJA_JJAS_precip_1900_2017.nc")

lat       = ncfile.lat.data # 90 to -90
lon       = ncfile.lon.data
time      = ncfile.time.data # 1891 -01 -01



f_01to55 = ncfile.sel(time=slice(1901, 1955))
ncfile.close()

jja_trend  = np.zeros((len(lat), len(lon)))
jja_p  = np.zeros((len(lat), len(lon)))
#print(jja_trend.shape)
for i in range(len(lat)):
    for j in range(len(lon)):
        if np.count_nonzero(~np.isnan(f_01to55['PRECT_JJA'].data[:, i, j])) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(1, 55, 55), f_01to55['PRECT_JJA'].data[:, i, j])
            slope, intercept, lower, upper = theilslopes(f_01to55['PRECT_JJA'].data[:, i, j], np.linspace(1, 55, 55))
            mk_result = mk.original_test(f_01to55['PRECT_JJA'].data[:, i, j])
            jja_trend[i, j]      = slope
            jja_p[i, j]      = p_value
        else:
            jja_p[i, j]      = np.nan
            jja_trend[i, j]  = np.nan


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

ncfile.attrs['description'] = 'Created on 2025-4-25 by /home/sun/uoe-code/paint/paint_ERL_figs2b_v4_UDEL_PRECT_linear_trend_250425.py.'
#
out_path = '/home/sun/data/download_data/data/analysis_data/'
ncfile.to_netcdf(out_path + 'Aerosol_Research_UDEL_PRECT_JJA_Theil_trend_1901to1955.nc')


#print(diff_precip_JJAS.shape) # (360, 720)

# ================================================================================

    



def paint_trend(lat, lon, diff, level, p, title_name, pic_path, pic_name):
    '''
        This function is plot the trend
    '''
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    from matplotlib import projections
    import cartopy.crs as ccrs

    # --- Set the figure ---
    proj    =  ccrs.PlateCarree()
    fig, ax =  plt.subplots(figsize=(20, 14), subplot_kw={'projection': proj})

    # --- Set range ---
    #lonmin,lonmax,latmin,latmax  =  65,93,5,35
    lonmin,lonmax,latmin,latmax  =  55,130,0,40
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(60,130,8,dtype=int),yticks=np.linspace(0,50,6,dtype=int),nx=1,ny=1,labelsize=30)

    # Shading for precipitation trend
    norm = BoundaryNorm(level, ncolors=256, clip=True)
    im  =  ax.contourf(lon, lat, diff, levels=level, cmap='coolwarm_r', alpha=1, extend='both', norm=norm)

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['.'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)
    ax.add_feature(cfeature.BORDERS, linewidth=1.)

    # Add a rectangle
    ax.add_patch(mpatches.Rectangle(xy=[72, 20], width=12, height=7.5,linestyle='--',
                                facecolor='none', edgecolor='grey', linewidth=3.5,
                                transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title(title_name, loc='left', fontsize=25)
    ax.set_title('GPCC', loc='right', fontsize=25)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.05, 0.05, 0.95, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=1, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=20)
    cb.set_ticks(level)  # 自定义刻度位置
    cb.set_ticklabels(level, fontsize=20)

    plt.savefig(pic_path + pic_name)
#

# =================== Calculation for period difference ==========================

data_path = '/home/sun/data/download_data/data/analysis_data/'

f0        = xr.open_dataset(data_path + 'Aerosol_Research_UDEL_PRECT_JJA_Theil_trend_1901to1955.nc')
#ref_file  = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc")
#f0        = f0.interp(lat=ref_file.lat.data, lon=ref_file.lon.data)

lat       = f0.lat.data # 90 to -90
lon       = f0.lon.data

f1        = f0.copy()

# ================== DONE for Calculation fot ttest ========================================

def main():
#    pvalue = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/Aerosol_research_GPCC_JJAS_periods_pvalue.nc")
    lev0 = np.array([-3, -2, -1, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 2, 3])
    f0['JJA_trend'].data[np.isnan(f0['JJA_trend'].data)] = 0
    f0['JJA_trend'].data = gaussian_filter(f0['JJA_trend'].data, sigma=1.)
    f0['JJA_trend'].data[np.isnan(f1['JJA_trend'].data)] = np.nan
    paint_trend(lat=lat, lon=lon, diff=f0['JJA_trend'].data * 50, level=lev0, p=f0['JJA_p'].data, title_name='1901-1955', pic_path='/home/sun/paint/ERL/', pic_name="ERL_figs2_v4_Aerosol_Research_UDEL_PRECT_JJAS_period_linear_trend_1901to1955.pdf")
#    paint_trend(lat=lat, lon=lon, diff=diff_precip_JJA,  level=np.linspace(-2., 2., 11), p=pvalue["pavlue_GPCC_JJAS_periods"], title_name='1936-1950', pic_path='/home/sun/data/download_data/paint/analysis_EU_aerosol_climate_effect/ERL/', pic_name="ERL_fig1b_v2_Aerosol_Research_GPCC_PRECT_JJA_period_diff_1901to1920_1936to1955.pdf")
#    paint_trend(lat=lat, lon=lon, diff=diff_precip_JJA, level=np.linspace(-3, 3, 13), p=None, title_name='JJA', pic_path='/home/sun/paint/aerosol_research/',    pic_name="Aerosol_Research_GPCC_PRECT_JJA_period_diff_1900_1960.pdf")
##
if __name__ == '__main__':
    main()