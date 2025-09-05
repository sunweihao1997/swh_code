'''
2025-8-25
This script is to calculate the gross correlation between the ship and ERA5 data
'''
import pandas as pd
import glob
import numpy as np

# 假设所有 CSV 文件都放在 ./data 文件夹
files = glob.glob("/mnt/f/ERA5_ship/donghai_ship_ERA5_QC/*.csv")

all_wind_speed = [] ; all_wind_direction = [] ; all_air_temperature = [] ; all_pressure = [] ; all_humidity = []
ERA5_wind_speed = [] ; ERA5_wind_direction = [] ; ERA5_air_temperature = [] ; ERA5_pressure = [] ; ERA5_humidity = []

for file in files:
    df = pd.read_csv(file)
    df = df.dropna()
    df = df[df["qc_overall"] == 1]

    df['ERA5_wind_speed'] = (df['ERA5_10m_u_component_of_wind']**2 + df['ERA5_10m_v_component_of_wind']**2)**0.5
    

    # Ship Observation Values
    values_wind_speed = df["真风速"].tolist()
    values_wind_direction = df["真风向"].tolist()
    values_air_temperature = df["气温"].tolist()
    values_pressure = df["气压"].tolist()
    values_humidity = df["湿度"].tolist()

    all_wind_speed.extend(values_wind_speed)
    all_wind_direction.extend(values_wind_direction)
    all_air_temperature.extend(values_air_temperature)
    all_pressure.extend(values_pressure)
    all_humidity.extend(values_humidity)

    # ERA5 Values
    values_ERA5_wind_speed = df["ERA5_wind_speed"].tolist()
    values_ERA5_wind_direction = df["ERA5_wind_direction"].tolist()
    values_ERA5_air_temperature = df["ERA5_2m_temperature"].tolist()
    values_ERA5_pressure = df["ERA5_mean_sea_level_pressure"].tolist()
    values_ERA5_humidity = df["ERA5_relative_humidity"].tolist()

    ERA5_wind_speed.extend(values_ERA5_wind_speed)
    ERA5_wind_direction.extend(values_ERA5_wind_direction)
    ERA5_air_temperature.extend(values_ERA5_air_temperature)
    ERA5_pressure.extend(values_ERA5_pressure)
    ERA5_humidity.extend(values_ERA5_humidity)

has_nan = np.isnan(ERA5_humidity).any()
print(has_nan)
    
# ========================== calculate correlation =========================
corr_matrix_windspeed = np.corrcoef(all_wind_speed, ERA5_wind_speed)
corr_np_ws = corr_matrix_windspeed[0, 1]
#print("相关系数 wind speed:", corr_np)

#corr_matrix_windspeed = np.corrcoef(all_wind_direction, ERA5_wind_direction)
#corr_np_ws = corr_matrix_windspeed[0, 1]
#print("相关系数 wind direction:", corr_np)

corr_matrix_windspeed = np.corrcoef(all_air_temperature, ERA5_air_temperature)
corr_np_at = corr_matrix_windspeed[0, 1]
#print("相关系数 air temperature:", corr_np)

corr_matrix_windspeed = np.corrcoef(all_pressure, ERA5_pressure)
corr_np_p = corr_matrix_windspeed[0, 1]
#print("相关系数 for pressure:", corr_np)

corr_matrix_windspeed = np.corrcoef(all_humidity, ERA5_humidity)
corr_np_h = corr_matrix_windspeed[0, 1]
#print("相关系数 for humidity:", corr_np)

# ========================== Plot scatter =========================
import matplotlib
matplotlib.use("Agg")             # 必须在 import pyplot 前调用
import matplotlib.pyplot as plt


# 创建 2x2 子图布局
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# 绘制第一个散点图
axs[0, 0].scatter(all_wind_speed, ERA5_wind_speed, marker='o', alpha=0.7)
axs[0, 0].set_title("Wind Speed", loc='left')
axs[0, 0].set_title(f"Correlation: {corr_np_ws:.2f}", loc='right')

# 绘制第二个散点图
axs[0, 1].scatter(all_air_temperature, ERA5_air_temperature, color='red', marker='x', alpha=0.7)
axs[0, 1].set_title("Air Temperature", loc='left')
axs[0, 1].set_title(f"Correlation: {corr_np_at:.2f}", loc='right')

# 绘制第三个散点图
axs[1, 0].scatter(all_pressure, ERA5_pressure, color='green', marker='^', alpha=0.7)
axs[1, 0].set_title("Pressure", loc='left')
axs[1, 0].set_title(f"Correlation: {corr_np_p:.2f}", loc='right')

# 绘制第四个散点图
axs[1, 1].scatter(all_humidity, ERA5_humidity, color='purple', marker='s', alpha=0.7)
axs[1, 1].set_title("Humidity", loc='left')
axs[1, 1].set_title(f"Correlation: {corr_np_h:.2f}", loc='right')

# 调整子图之间的间距
plt.tight_layout()
plt.savefig("/mnt/f/ERA5_ship/donghai_ship_ERA5_QC/correlation_scatter.png", dpi=300)