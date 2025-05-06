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

module_path = '/home/sun/uoe-code/module/'
sys.path.append(module_path)
from module_sun import *

# =================== File Location =============================

file_path = '/home/sun/data/download_data/data/download/GPCC_NCEP_prect/'
file_name = 'precip.mon.total.1x1.v2020.nc'

f0        = xr.open_dataset(file_path + file_name)
#print(f0)

#print(time)
varname   = 'precip'

# ===============================================================

# Interpolate into the CESM resolution
ref_file  = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/CESM_PRECT_BTAL_BTALnEU_JJA_JJAS_1850_2006.nc")
#f0        = f0.interp(lat=ref_file.lat.data, lon=ref_file.lon.data)

lat       = f0.lat.data # 90 to -90
lon       = f0.lon.data
time      = f0.time.data # 1891 -01 -01


# ================== Calculation for JJA / JJAS precipitation ================

month0 = [6, 7, 8, 9] # JJAS
month1 = [6, 7, 8] # JJA

avg_JJAS = np.zeros((129, len(lat), len(lon)))
avg_JJA  = np.zeros((129, len(lat), len(lon)))

f0_JJAS  = f0.sel(time=f0.time.dt.month.isin(month0)) ; print(len(f0_JJAS.time.data)/len(month0))
f0_JJA   = f0.sel(time=f0.time.dt.month.isin(month1)) ; print(len(f0_JJA.time.data)/len(month1))

for mm in range(129):
    avg_JJAS[mm] = np.average(f0_JJAS[varname].data[mm * len(month0) : (mm * len(month0) + len(month0))], axis=0)
    avg_JJA[mm]  = np.average(f0_JJA[varname].data[mm * len(month1)  : (mm * len(month1) + len(month1))], axis=0)

print('Both JJAS and JJA has been calculated!')

ncfile  =  xr.Dataset(
    {
        "JJAS_PRECT": (["time", "lat", "lon"], avg_JJAS/31),
        "JJA_PRECT":  (["time", "lat", "lon"], avg_JJA/31),
    },
    coords={
        "time": (["time"], np.linspace(1891, 1891 + 128, 129)),
        "lat":  (["lat"],  lat),
        "lon":  (["lon"],  lon),
    },
    )

ncfile["JJAS_PRECT"].attrs['units'] = 'mm day^-1'
ncfile["JJA_PRECT"].attrs['units']  = 'mm day^-1'

ncfile.attrs['description'] = 'Created on 2024-1-8.'
ncfile.attrs['script'] = 'paint_ERL_fig1_GPCC_PRECT_period_difference_1940_1960_231221.py on UOE'
#
out_path = '/home/sun/data/download_data/data/analysis_data/'
ncfile.to_netcdf(out_path + 'Aerosol_Research_GPCC_PRECT_JJA_JJAS_average.nc')

# ================================================================================

# ================ Calculation for 1901to1955 linear trend =======================
f_01to55 = ncfile.sel(time=slice(1901, 1955))
ncfile.close()

jja_trend  = np.zeros((len(lat), len(lon)))
jjas_trend = np.zeros((len(lat), len(lon)))
jja_p  = np.zeros((len(lat), len(lon)))
jjas_p = np.zeros((len(lat), len(lon)))
#print(jja_trend.shape)
for i in range(len(lat)):
    for j in range(len(lon)):
        slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(1, 55, 55), f_01to55['JJA_PRECT'].data[:, i, j])
        jja_trend[i, j]  = slope
        jja_p[i, j]      = p_value

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.linspace(1, 55, 55), f_01to55['JJAS_PRECT'].data[:, i, j])
        jjas_trend[i, j] = slope
        jjas_p[i, j]     = p_value

# Write trend into file
ncfile  =  xr.Dataset(
    {
        "JJAS_trend": (["lat", "lon"], jjas_trend),
        "JJA_trend":  (["lat", "lon"], jja_trend),
        "JJAS_p": (["lat", "lon"], jjas_p),
        "JJA_p":  (["lat", "lon"], jja_p),
    },
    coords={
        "lat":  (["lat"],  lat),
        "lon":  (["lon"],  lon),
    },
    )

ncfile["JJAS_trend"].attrs['units'] = 'mm day^-1 year^-1'
ncfile["JJA_trend"].attrs['units']  = 'mm day^-1 year^-1'

ncfile.attrs['description'] = 'Created on 2024-7-7.'
ncfile.attrs['script'] = 'paint_ERL_fig1_modified_GPCC_PRECT_period_difference_1940_1960_v2_240707.py on UOE'
#
out_path = '/home/sun/data/download_data/data/analysis_data//'
ncfile.to_netcdf(out_path + 'Aerosol_Research_GPCC_PRECT_JJA_JJAS_linear_trend_1901to1955.nc')


#print(diff_precip_JJAS.shape) # (360, 720)

# ================================================================================

def cal_ttest(data1, data2):
    '''
        This function calculate difference using ttest
        data1 and data2 should be anomaly value compared with climatology
    '''
    import scipy.stats as stats

    # claim the array to save the p value
    p_array = np.zeros((data1.shape[1], data1.shape[2])) # it should be (lat, lon)

    # Compare on each point
    y_num = data1.shape[1]  ;  x_num = data1.shape[2]
    for yy in range(y_num):
        for xx in range(x_num):
            t_value, p_value = stats.ttest_ind(data1[:, yy, xx], data2[:, yy, xx])
            print(p_value)
            p_array[yy, xx] = p_value

    return p_array
    



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
    lonmin,lonmax,latmin,latmax  =  60,125,5,35
    extent     =  [lonmin,lonmax,latmin,latmax]

    # --- Tick setting ---
    set_cartopy_tick(ax=ax,extent=extent,xticks=np.linspace(50,140,7,dtype=int),yticks=np.linspace(10,60,6,dtype=int),nx=1,ny=1,labelsize=25)

    # Shading for precipitation trend
    im  =  ax.contourf(lon, lat, diff, levels=level, cmap='coolwarm_r', alpha=1, extend='both')

#    # Stippling picture
    sp  =  ax.contourf(lon, lat, p, levels=[0., 0.1], colors='none', hatches=['.'])

    # --- Coast Line ---
    ax.coastlines(resolution='50m', lw=1.5)

    # Add a rectangle
    ax.add_patch(mpatches.Rectangle(xy=[72, 20], width=12, height=7.5,linestyle='--',
                                facecolor='none', edgecolor='grey', linewidth=3.5,
                                transform=ccrs.PlateCarree()))

    # --- title ---
    ax.set_title(title_name, loc='left', fontsize=25)
    ax.set_title('GPCC', loc='right', fontsize=25)

    # ========= add colorbar =================
    fig.subplots_adjust(top=0.8) 
    cbar_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig.colorbar(im, cax=cbar_ax, shrink=0.5, pad=0.01, orientation='horizontal')
    #cb.ax.set_xticks(levels)
    cb.ax.tick_params(labelsize=20)

    plt.savefig(pic_path + pic_name)
#

# =================== Calculation for period difference ==========================

data_path = '/home/sun/data/download_data/data/analysis_data/'

f0        = xr.open_dataset(data_path + 'Aerosol_Research_GPCC_PRECT_JJA_JJAS_linear_trend_1901to1955.nc')

lat       = f0.lat.data ; lon       = f0.lon.data


# ================== DONE for Calculation fot ttest ========================================

def main():
#    pvalue = xr.open_dataset("/home/sun/data/download_data/data/analysis_data/Aerosol_research_GPCC_JJAS_periods_pvalue.nc")
    paint_trend(lat=lat, lon=lon, diff=f0['JJAS_trend'].data * 10, level=np.linspace(-0.5, .5, 11), p=f0['JJA_p'].data, title_name='1901-1955', pic_path='/home/sun/paint/ERL/', pic_name="ERL_fig1a_v4_Aerosol_Research_GPCC_PRECT_JJA_period_linear_trend_1901to1955.pdf")
#    paint_trend(lat=lat, lon=lon, diff=diff_precip_JJA,  level=np.linspace(-2., 2., 11), p=pvalue["pavlue_GPCC_JJAS_periods"], title_name='1936-1950', pic_path='/home/sun/data/download_data/paint/analysis_EU_aerosol_climate_effect/ERL/', pic_name="ERL_fig1b_v2_Aerosol_Research_GPCC_PRECT_JJA_period_diff_1901to1920_1936to1955.pdf")
#    paint_trend(lat=lat, lon=lon, diff=diff_precip_JJA, level=np.linspace(-3, 3, 13), p=None, title_name='JJA', pic_path='/home/sun/paint/aerosol_research/',    pic_name="Aerosol_Research_GPCC_PRECT_JJA_period_diff_1900_1960.pdf")
##
if __name__ == '__main__':
    main()