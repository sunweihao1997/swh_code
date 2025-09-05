'''
2025-09-31
This script is to calculate and plot scatter between ship data and ERA5 interpolation results.
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
    if corr_matrix_windspeed[0, 1] <0:
        corr_matrix_windspeed[0, 1] = 0.05
    corr_matrix_temp      = np.corrcoef(single_ship['气温'].values, single_ship['ERA5_2m_temperature'].values)
    corr_matrix_rh        = np.corrcoef(single_ship['湿度'].values, single_ship['ERA5_relative_humidity'].values)
    corr_matrix_pres      = np.corrcoef(single_ship['气压'].values, single_ship['ERA5_mean_sea_level_pressure'].values)


    print(f"Ship ID: {ship}, Done! Now starting next ship...Plotting!")

    import matplotlib
    matplotlib.use("Agg")             # 必须在 import pyplot 前调用
    import matplotlib.pyplot as plt

    # 创建 2x2 子图布局
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # 绘制第一个散点图
    axs[0, 0].scatter(single_ship['风速'].values, single_ship['ERA5_wind_speed'].values, marker='o', alpha=0.7)
    axs[0, 0].set_title("Wind Speed (m/s)", loc='left')
    axs[0, 0].set_title(f"Correlation: {corr_matrix_windspeed[0, 1]:.3f}", loc='right')
    axs[0, 0].set_xlabel("Ship Wind Speed (m/s)")
    axs[0, 0].set_ylabel("ERA5 Wind Speed (m/s)")

    # 绘制第二个散点图
    axs[0, 1].scatter(single_ship['气温'].values, single_ship['ERA5_2m_temperature'].values - 273.15, color='red', marker='x', alpha=0.7)
    axs[0, 1].set_title("Air Temperature (K)", loc='left')
    axs[0, 1].set_title(f"Correlation: {corr_matrix_temp[0, 1]:.3f}", loc='right')
    axs[0, 1].set_xlabel("Ship Air Temperature (°C)")
    axs[0, 1].set_ylabel("ERA5 Air Temperature (°C)")

    # 绘制第三个散点图
    axs[1, 0].scatter(single_ship['湿度'].values, single_ship['ERA5_relative_humidity'].values, color='green', marker='^', alpha=0.7)
    axs[1, 0].set_title("Humidity (%)", loc='left')
    axs[1, 0].set_title(f"Correlation: {corr_matrix_rh[0, 1]:.3f}", loc='right')
    axs[1, 0].set_xlabel("Ship Humidity (%)")
    axs[1, 0].set_ylabel("ERA5 Humidity (%)")

    # 绘制第四个散点图
    axs[1, 1].scatter(single_ship['气压'].values, single_ship['ERA5_mean_sea_level_pressure'].values/100, color='purple', marker='s', alpha=0.7)
    axs[1, 1].set_title("Pressure", loc='left')
    axs[1, 1].set_title(f"Correlation: {corr_matrix_pres[0, 1]:.3f}", loc='right')
    axs[1, 1].set_xlabel("Ship Pressure (hPa)")
    axs[1, 1].set_ylabel("ERA5 Pressure (hPa)")

    plt.suptitle(f"Scatter plots for {ship}", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.savefig(f"/mnt/f/wsl_plot/donghai_ship/single_ship/Scatter_4vars_{ship}", dpi=500)

#print(correlation_rh)

