'''
2023-10-25
This script plot the evolution of the rain belt among the tropical Asian

Here I need to set a range of longitude to see the month/latitude evolution
To meet my goals, I will plot the following pictures
1. BTAL and BTALnEU experiments for their changes in precipitation evolution during the same periods (1901-1921) to (1941-1961)
2. The difference between the above teo experiments to show the influence of the EU aerosol on the advancement of Indian Monsoon

2023-12-07 update:
1. change the function cal_period_average
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

# ========== Define the period for calculating average ==========
periodA_1 = 50 ; periodA_2 = 70
periodB_1 = 90 ; periodB_2 = 110

# Define the longitude range
range1 = 70 ; range2 = 90

#levels = np.array([-1.5, -1.2, -0.9, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.2, 1.5,])
levels = np.linspace(3, 24, 22)

class calculation:
    def cal_period_average(data, dim_num, start, end, varname):
        '''This function is to calculate the monthly-averaged data'''
        # 1. Claim the array data regarding its dimension
        if dim_num == 2:
            avg_period = np.zeros((12, 96, 144))
        else:
            avg_period = np.zeros((12, 29, 96, 144))

        # Only select the data in the start:end peiod
        #print(data)
        data = data.sel(time=slice(str(start) + "-01-01", str(end) + "-12-31"))

        for mmmm in range(12):
            data_this_month = data.sel(time=data.time.dt.month.isin([int(mmmm + 1)]))

            month_num = len(data_this_month['time'].data)
            for yyyy in range(month_num):
                avg_period[mmmm] += (data_this_month[varname].data[yyyy] / month_num)

        print("successfully calculate the period average for every month")

        return avg_period

def paint_rain_evolution_diff(ncfile, varname, name):
    '''This function is to plot the evolution of the rainy season'''
    ax         =  plt.subplot()

    ncfile1    =  ncfile.sel(lon=slice(range1, range2))
    print(np.average(ncfile1[varname].data, axis=2).shape)

    im  =  ax.contourf(np.linspace(1, 12, 12), ncfile1['lat'].data, np.swapaxes(np.average(ncfile1[varname].data, axis=2), 0, 1) * 86400 * 1000, levels, cmap='Blues', alpha=1, extend='both')

    ax.set_xticks(np.linspace(1, 12, 12))
    ax.set_yticks(np.linspace(-10, 40, 6))
    ax.set_ylim((-10, 45))

    ax.set_ylabel("Influence of EU emission", fontsize=12.5)
    ax.set_title("BTAL - BTALnEU")

    plt.colorbar(im, orientation='horizontal')
    plt.savefig("/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/rain_evolution/" + name)
    

def main():
    # ================ 1. Calculate period-average =====================
    prect1  =  xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/noEU_precipitation_ensemble_mean_231004.nc")
    prect0  =  xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_precipitation_ensemble_mean_231005.nc")
    # Please note: The total precipitation is the add of PRECL and PRECC
#    a = calculation.cal_period_average(data=prect0['PRECC'], dim_num=2, start=1900, end=1920, varname='PRECC')

    con_c1 = calculation.cal_period_average( data=prect0, dim_num=2, start=1900, end=1920, varname='PRECC')
    con_l1 = calculation.cal_period_average( data=prect0, dim_num=2, start=1900, end=1920, varname='PRECL')
    neu_c1 = calculation.cal_period_average( data=prect1, dim_num=2, start=1900, end=1920, varname='PRECC')
    neu_l1 = calculation.cal_period_average( data=prect1, dim_num=2, start=1900, end=1920, varname='PRECL')

    con_c2 = calculation.cal_period_average( data=prect0, dim_num=2, start=1940, end=1960, varname='PRECC')
    con_l2 = calculation.cal_period_average( data=prect0, dim_num=2, start=1940, end=1960, varname='PRECL')
    neu_c2 = calculation.cal_period_average( data=prect1, dim_num=2, start=1940, end=1960, varname='PRECC')
    neu_l2 = calculation.cal_period_average( data=prect1, dim_num=2, start=1940, end=1960, varname='PRECL')

    # ============== 2. Calculate the long-term change ================
    avg_p1_con = con_c1 + con_l1 ; avg_p2_con = con_c2 + con_l2
    avg_p1_neu = neu_c1 + neu_l1 ; avg_p2_neu = neu_c2 + neu_l2

    diff_p1p2_con = avg_p2_con - avg_p1_con
    diff_p1p2_neu = avg_p2_neu - avg_p1_neu

    diff_con_neu  = diff_p1p2_con - diff_p1p2_neu
    print("Oh Ye")

    # ============== 3. Combine them to the ncfile ====================
    ncfile  =  xr.Dataset(
            {
                "avg_p2_con": (["time", "lat", "lon"], avg_p2_con),
                "avg_p2_neu": (["time", "lat", "lon"], avg_p2_neu),
            },
            coords={
                "time": (["time"], np.linspace(1, 12, 12, dtype=int)),
                "lat":  (["lat"],  prect0['lat'].data),
                "lon":  (["lon"],  prect0['lon'].data),
            },
            )
    
    paint_rain_evolution_diff(ncfile, "avg_p2_con", "BTAL_hovmoller_1940_1960_average.pdf")
    paint_rain_evolution_diff(ncfile, "avg_p2_neu", "BTALnEU_hovmoller_1940_1960_average.pdf")

    

if __name__ == '__main__':
    main()