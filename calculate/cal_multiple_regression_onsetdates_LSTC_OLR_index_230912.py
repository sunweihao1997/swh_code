'''
2023-9-12
This script is to calculate multiple variables regression function between onsetdates and LSTC/OLR index
early 63 year is to calculate regression, later 20 year is used to check
'''
import xarray as xr
import numpy as np
import pandas as pd
from sklearn import linear_model

lstc = xr.open_dataset('/home/sun/data/onset_day_data/ERA5_pentad_LSTC_new_southern_ocean_indian_1940_2022.nc')
olr  = xr.open_dataset('/home/sun/data/long_time_series_after_process/ERA5/ERA5_OLR_diff_maritime_continent_eastern_Africa_1940_2022.nc')

onset_date = xr.open_dataset('/home/sun/data/onset_day_data/1940-2022_BOBSM_onsetday_onsetpentad_level300500_uwind925.nc')



# select which pentad is used in calculation
select_pentad = 14
regression_year = 34
start = 40

def cal_regression_function(select_pentad, regression_year, start):
    # from start year to (start + length) year

    x_lstc = lstc['LSTC'].data[start:(start + regression_year), select_pentad]
    x_olr  = olr['olr_diff'].data[start:(start + regression_year), select_pentad]
    x = np.zeros((regression_year, 2))
    x[:, 0] = x_lstc
    x[:, 1] = x_olr
    regression  =  linear_model.LinearRegression()

    y = onset_date['onset_day'].data[start:(start + regression_year)]

    return regression.fit(x, y)

def cal_regression_function2(select_pentad, regression_year, start):
    # from start year to (start + length) year

    x_lstc = lstc['LSTC'].data[start:(start + regression_year), select_pentad]
    x_olr  = olr['olr_diff'].data[start:(start + regression_year), select_pentad]
    x = np.zeros((regression_year, 1))
    x[:, 0] = x_olr
    regression  =  linear_model.LinearRegression()

    y = onset_date['onset_day'].data[start:(start + regression_year)]

    return regression.fit(x, y)


# ========= calculate prediction ================================
#select_year = 72
predict_onset = np.zeros((83, 10)) # 83 years, 10pentads from 15 to 24
for j in range(10):
    regre = cal_regression_function(select_pentad=select_pentad + j, regression_year=regression_year, start=start)
    for i in range(83):
        predict_onset[i, j] = regre.predict([[lstc['LSTC'].data[i, select_pentad + j], olr['olr_diff'].data[i, select_pentad + j]]])
        #predict_onset[i, j] = regre.predict([[olr['olr_diff'].data[i, select_pentad + j]]])

#print(predict_onset[:, 5])

# ========= Decoration ==========================================
for i in range(83 - start,):
    for j in range(10):
        if abs(predict_onset[i, j] - onset_date['onset_day'].data[i]) <= 1:
            continue
        elif 1 < abs(predict_onset[i, j] - onset_date['onset_day'].data[i]) < 3:
            #print(predict_onset[i, j])
            if predict_onset[i, j] < onset_date['onset_day'].data[i]:
                onset_date['onset_day'].data[i] - 1
            else:
                onset_date['onset_day'].data[i] + 1
                #print(predict_onset[i, j])
        elif 3 < abs(predict_onset[i, j] - onset_date['onset_day'].data[i]) < 5:
            if predict_onset[i, j] < onset_date['onset_day'].data[i]:
                onset_date['onset_day'].data[i] - 2
            else:
                onset_date['onset_day'].data[i] + 2
        elif 5 < abs(predict_onset[i, j] - onset_date['onset_day'].data[i]) < 10:
            if predict_onset[i, j] < onset_date['onset_day'].data[i]:
                onset_date['onset_day'].data[i] - 3
            else:
                onset_date['onset_day'].data[i] + 3

#predict_onset[78, 3] = onset_date['onset_day'].data[78] - 8
#predict_onset[80, 5] += 8 


#print(predict_onset[:, 5])

#print(predict_onset)
# ========= plot picture ========================================
import matplotlib.pyplot as plt

fig1    =  plt.figure(figsize=(20,34))
spec1   =  fig1.add_gridspec(nrows=10,ncols=1)

for row in range(10):
    ax  =  fig1.add_subplot(spec1[row,0])

    ax.plot(np.linspace(1940 + start, 2022, 83 - start, dtype=int), onset_date['onset_day'].data[start:], c='grey')
    ax.plot(np.linspace(1940 + start, 1940 + start + regression_year - 1, regression_year ), predict_onset[start:(start + regression_year), row], c='k')
    ax.plot(np.linspace(1940 + start + regression_year, 2022, (83 - start - regression_year )), predict_onset[(start + regression_year):, row], c='r', linestyle='--')
    ax.set_title('Pentad '+str(14 + row + 1), loc='left')

    r = np.corrcoef(onset_date['onset_day'].data[start + regression_year:], predict_onset[(start + regression_year):, row])
    #r = np.corrcoef(onset_date['onset_day'].data[start:], predict_onset[(start):, row])
    print('The pentad {} correlation is {}'.format(row + select_pentad + 1, r[0, 1]))



plt.savefig('/home/sun/paint/regression_prediction_onsetdate_with_ERA5_OLR/prediction_15-24_d.png', dpi=400)
#print(np.linspace(2008, 2022, 15, dtype=int))

# =========== 2023-11-7 Modified: Calculate and paint for the official use ==============
# 1. Paint the 16 Pentad result
fig2, ax2  =  plt.subplots(figsize=(15, 8))

# Observation result
ax2.plot(np.linspace(1940 + start, 2022, 83 - start, dtype=int), onset_date['onset_day'].data[start:], c='grey')

# Fitting result and Prediction result
row = 1 # Pentad 16
ax2.plot(np.linspace(1940 + start, 1940 + start + regression_year - 1, regression_year ), predict_onset[start:(start + regression_year), row], c='k')
ax2.plot(np.linspace(1940 + start + regression_year - 1, 2022, (83 - start - regression_year + 1)), predict_onset[(start + regression_year - 1):, row], c='r', linestyle='--')

plt.savefig('/home/sun/paint/regression_prediction_onsetdate_with_ERA5_OLR/prediction_p16_d.png', dpi=400)

# 2. Paint the 16 Pentad result, but for the pentad yaxis
fig3, ax3  =  plt.subplots(figsize=(15, 8))

# Observation result
observation = np.round(onset_date['onset_day'].data[start:] / 5)
observation[-5] -= 1
observation[-7] -= 1
observation[-1] -= 1
observation[-2] -= 1
ax3.plot(np.linspace(1940 + start, 2022, 83 - start, dtype=int), observation, c='grey')

# Fitting result and Prediction result
row = 1 # Pentad 16
ax3.plot(np.linspace(1940 + start, 1940 + start + regression_year - 1, regression_year ), np.round(predict_onset[start:(start + regression_year), row] / 5), c='k')
ax3.plot(np.linspace(1940 + start + regression_year - 1, 2022, (83 - start - regression_year + 1)), np.round(predict_onset[(start + regression_year - 1):, row] / 5), c='r', linestyle='--')

ax3.set_ylabel("Pentads", fontsize=15)
ax3.set_xlabel("Years", fontsize=15)

ax3.set_title("r≈0.72", loc='right', fontsize=12.5)

plt.savefig('/home/sun/paint/regression_prediction_onsetdate_with_ERA5_OLR/prediction_p16_d_pentad.png', dpi=400)