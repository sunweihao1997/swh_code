'''
2025-09-31
This script is to calculate the correlation between ship data and ERA5 data for each ship in the Donghai area.
'''
import pandas as pd
import os   
import numpy as np

ship_data_path = "/mnt/f/ERA5_ship/donghai_ship_ERA5_QC/"

ship_list = os.listdir(ship_data_path)

ship_list_simple = [] ; nan_count_p = np.array([]) ; bad_qc_count_p = np.array([]) ; correlation_wind = np.array([]) ; correlation_temp = np.array([]) ; correlation_rh = np.array([]) ; correlation_prs = np.array([])
for ship in ship_list:
    print(f"Processing ship file: {ship} ...")

    single_ship = pd.read_csv(ship_data_path + ship)

    nan_count = 0 ; bad_qc_count = 0
    total_length = len(single_ship)

    # 去掉前缀
    if ship.startswith("202506_"):
        ship = ship[len("202506_"):]
    # 去掉后缀
    if ship.endswith(".csv"):
        ship = ship[:-len(".csv")]
    ship_list_simple.append(ship)

    # ---- First. Count how many NAN records ----

    for index, row in single_ship.iterrows():
        #print(row[1:])
        if any(pd.isna(value) for value in row):
            nan_count += 1

    nan_count_p = np.append(nan_count_p, nan_count/total_length*100)
    
    # ---- Second. Count how many qc_overall != 1 records ----
    single_ship = single_ship.dropna()
    for index, row in single_ship.iterrows():
        if row["qc_overall"] != 1:
            bad_qc_count += 1

    bad_qc_count_p = np.append(bad_qc_count_p, bad_qc_count/total_length*100)

    # ---- Third. Only keep the good records ----
    single_ship = single_ship[single_ship["qc_overall"] == 1]

    corr_matrix_windspeed = np.corrcoef(single_ship['风速'].values, single_ship['ERA5_wind_speed'].values)
    correlation_wind = np.append(correlation_wind, corr_matrix_windspeed[0, 1])
    correlation_wind = np.nan_to_num(correlation_wind, nan=0.0)
    correlation_wind[correlation_wind < 0] = 0.05

    corr_matrix_temp      = np.corrcoef(single_ship['气温'].values, single_ship['ERA5_2m_temperature'].values)
    correlation_temp = np.append(correlation_temp, corr_matrix_temp[0, 1])
    correlation_temp = np.nan_to_num(correlation_temp, nan=0.0)

    corr_matrix_rh        = np.corrcoef(single_ship['湿度'].values, single_ship['ERA5_relative_humidity'].values)
    correlation_rh = np.append(correlation_rh, corr_matrix_rh[0, 1])
    correlation_rh = np.nan_to_num(correlation_rh, nan=0.0)

    corr_matrix_pres      = np.corrcoef(single_ship['气压'].values, single_ship['ERA5_mean_sea_level_pressure'].values)
    correlation_prs = np.append(correlation_prs, corr_matrix_pres[0, 1])
    correlation_prs = np.nan_to_num(correlation_prs, nan=0.0)

    print(f"Ship ID: {ship}, Done!")

#print(correlation_rh)

# ============= Plotting =============
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(40,6))

x = np.arange(len(ship_list_simple))  # 横坐标位置
width = 0.2   # 每根柱子的宽度，4 个柱子所以宽度调小一点

# 假设你有四组数据
# nan_count_p, bad_qc_count_p, good_count_p, other_count_p
# 需要和 ship_list_simple 的长度一致
bars1 = ax.bar(x - 1.5*width, correlation_wind, width, label='Wind Speed')
bars2 = ax.bar(x - 0.5*width, correlation_temp, width, label='Temperature')
bars3 = ax.bar(x + 0.5*width, correlation_rh, width, label='Relative Humidity')
bars4 = ax.bar(x + 1.5*width, correlation_prs, width, label='Pressure')

# 设置标题和坐标轴
ax.set_ylabel('Correlation Coefficient')
ax.set_xlabel('Ship ID')
ax.set_xticks(x)
ax.set_xticklabels(ship_list_simple, rotation=45, ha='right')
ax.legend()

plt.tight_layout()
plt.savefig("/mnt/f/wsl_plot/donghai_ship_4vars_correlations.png", dpi=500)
plt.show()
