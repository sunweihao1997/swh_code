'''
2025-09-31
This script is to calculate and plot scatter between ship data and ERA5 interpolation results.
'''
import pandas as pd
import os   
import numpy as np
from datetime import datetime

ship_data_path = "/mnt/f/ERA5_ship/donghai_ship_ERA5_QC/"

ship_list = os.listdir(ship_data_path)

ship_list_simple = [] ; nan_count_p = np.array([]) ; bad_qc_count_p = np.array([]) ; correlation_wind = np.array([]) ; correlation_temp = np.array([]) ; correlation_rh = np.array([]) ; correlation_prs = np.array([])
for ship in ship_list:
    if ship == "202506_D5DA8.csv":

        continue
        
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

    single_ship["time"] = pd.to_datetime(
        single_ship[["年", "月", "日", "时"]].astype(str).agg("-".join, axis=1),
        format="%Y-%m-%d-%H"
    )

    if ship == "D5DA8":
        continue


    import matplotlib
    matplotlib.use("Agg")             # 必须在 import pyplot 前调用
    import matplotlib.pyplot as plt

    # 创建 4x1 子图布局
    fig, axs = plt.subplots(4, 1, figsize=(20, 10))


    axs[0].plot(single_ship["time"], single_ship['风速'].values, marker='o', alpha=0.7, label='Ship Wind Speed', color='blue')
    axs[0].plot(single_ship["time"], single_ship['ERA5_wind_speed'].values, marker='^', alpha=0.7, label='ERA5 Wind Speed', color='lightblue')
    axs[0].set_ylabel("Wind Speed (m/s)")
    axs[0].set_xlabel("Time")
    axs[0].legend(loc='upper left')

    axs[1].plot(single_ship["time"], single_ship['气温'].values, marker='o', alpha=0.7, label='Ship Air Temperature', color='red')
    axs[1].plot(single_ship["time"], single_ship['ERA5_2m_temperature'].values - 273.15, marker='^', alpha=0.7, label='ERA5 Air Temperature', color='salmon')
    axs[1].set_ylabel("Air Temperature (°C)")
    axs[1].set_xlabel("Time")
    axs[1].legend(loc='upper left')

    axs[2].plot(single_ship["time"], single_ship['湿度'].values, marker='o', alpha=0.7, label='Ship Humidity', color='green')
    axs[2].plot(single_ship["time"], single_ship['ERA5_relative_humidity'].values, marker='^', alpha=0.7, label='ERA5 Humidity', color='lightgreen')
    axs[2].set_ylabel("Humidity (%)")
    axs[2].set_xlabel("Time")
    axs[2].legend(loc='upper left')

    axs[3].plot(single_ship["time"], single_ship['气压'].values, marker='o', alpha=0.7, label='Ship Pressure', color='purple')
    axs[3].plot(single_ship["time"], single_ship['ERA5_mean_sea_level_pressure'].values/100, marker='^', alpha=0.7, label='ERA5 Pressure', color='violet')
    axs[3].set_ylabel("Pressure (hPa)")
    axs[3].set_xlabel("Time")
    axs[3].legend(loc='upper left')


    plt.suptitle(f"Scatter plots for {ship}", fontsize=16, y=0.99)
    plt.tight_layout()
    plt.savefig(f"/mnt/f/wsl_plot/donghai_ship/single_ship/zhexian_plot/Line_4vars_{ship}", dpi=500)

#print(correlation_rh)

