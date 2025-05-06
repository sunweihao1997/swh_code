'''
2023-10-25
This script plot the evolution of the rain belt among the tropical Asian

Here I need to set a range of longitude to see the month/latitude evolution
To meet my goals, I will plot the following pictures
1. BTAL and BTALnEU experiments for their changes in precipitation evolution during the same periods (1901-1921) to (1941-1961)
2. The difference between the above teo experiments to show the influence of the EU aerosol on the advancement of Indian Monsoon
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

levels = np.array([-1.5, -1.2, -0.9, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9, 1.2, 1.5,])

class calculation:
    def cal_period_average(data, dim_num, start, end):
        '''This function is to calculate the monthly-averaged data'''
        # 1. Claim the array data regarding its dimension
        if dim_num == 2:
            avg_period = np.zeros((12, 96, 144))
        else:
            avg_period = np.zeros((12, 29, 96, 144))

        for yyyy in range(start, end):
            for mmmm in range(12):
                avg_period[mmmm] += data[yyyy * 12 + mmmm] / (end - start)

        print("successfully calculate the period average for every month")

        return avg_period

def paint_rain_evolution_diff(ncfile, varname, name):
    '''This function is to plot the evolution of the rainy season'''
    ax         =  plt.subplot()

    ncfile1    =  ncfile.sel(lon=slice(range1, range2))
    print(np.average(ncfile1[varname].data, axis=2).shape)

    im  =  ax.contourf(np.linspace(1, 12, 12), ncfile1['lat'].data, np.swapaxes(np.average(ncfile1[varname].data, axis=2), 0, 1) * 86400 * 1000, levels, cmap='bwr_r', alpha=1, extend='both')

    ax.set_xticks(np.linspace(1, 12, 12))
    ax.set_yticks(np.linspace(-10, 40, 6))
    ax.set_ylim((-10, 45))

    plt.colorbar(im, orientation='horizontal')
    plt.savefig("/exports/csce/datastore/geos/users/s2618078/paint/analysis_EU_aerosol_climate_effect/rain_evolution/" + name)
    

def main():
    # ================ 1. Calculate period-average =====================
    prect1  =  xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/noEU_precipitation_ensemble_mean_231004.nc")
    prect0  =  xr.open_dataset("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_precipitation_ensemble_mean_231005.nc")
    # Please note: The total precipitation is the add of PRECL and PRECC
    with concurrent.futures.ProcessPoolExecutor() as executor:
        con_c1 = executor.submit(calculation.cal_period_average, data=prect0['PRECC'].data, dim_num=2, start=periodA_1, end=periodA_2)
        con_l1 = executor.submit(calculation.cal_period_average, data=prect0['PRECL'].data, dim_num=2, start=periodA_1, end=periodA_2)
        neu_c1 = executor.submit(calculation.cal_period_average, data=prect1['PRECC'].data, dim_num=2, start=periodA_1, end=periodA_2)
        neu_l1 = executor.submit(calculation.cal_period_average, data=prect1['PRECL'].data, dim_num=2, start=periodA_1, end=periodA_2)

        con_c2 = executor.submit(calculation.cal_period_average, data=prect0['PRECC'].data, dim_num=2, start=periodB_1, end=periodB_2)
        con_l2 = executor.submit(calculation.cal_period_average, data=prect0['PRECL'].data, dim_num=2, start=periodB_1, end=periodB_2)
        neu_c2 = executor.submit(calculation.cal_period_average, data=prect1['PRECC'].data, dim_num=2, start=periodB_1, end=periodB_2)
        neu_l2 = executor.submit(calculation.cal_period_average, data=prect1['PRECL'].data, dim_num=2, start=periodB_1, end=periodB_2)

    # ============== 2. Calculate the long-term change ================
    avg_p1_con = con_c1.result() + con_l1.result() ; avg_p2_con = con_c2.result() + con_l2.result()
    avg_p1_neu = neu_c1.result() + neu_l1.result() ; avg_p2_neu = neu_c2.result() + neu_l2.result()

    diff_p1p2_con = avg_p2_con - avg_p1_con
    diff_p1p2_neu = avg_p2_neu - avg_p1_neu

    diff_con_neu  = diff_p1p2_con - diff_p1p2_neu
    print("Oh Ye")

    # ============== 3. Combine them to the ncfile ====================
    ncfile  =  xr.Dataset(
            {
                "diff_p1p2_con": (["time", "lat", "lon"], diff_p1p2_con),
                "diff_p1p2_neu": (["time", "lat", "lon"], diff_p1p2_neu),
                "diff_con_neu":  (["time", "lat", "lon"], diff_con_neu),
            },
            coords={
                "time": (["time"], np.linspace(1, 12, 12, dtype=int)),
                "lat":  (["lat"],  prect0['lat'].data),
                "lon":  (["lon"],  prect0['lon'].data),
            },
            )
    
    paint_rain_evolution_diff(ncfile, "diff_con_neu", "BTAL_BTALnEU_diff.pdf")
    paint_rain_evolution_diff(ncfile, "diff_p1p2_con", "BTAL.pdf")
    paint_rain_evolution_diff(ncfile, "diff_p1p2_neu", "BTALnEU.pdf")

    




if __name__ == '__main__':
    main()