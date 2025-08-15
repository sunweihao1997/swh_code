import cdsapi
import os
from netCDF4 import Dataset

def is_file_complete(filepath):
    """
    检查 NetCDF 文件是否完整。
    尝试打开文件并读取其基本信息，如果成功则认为文件完整。
    """
    if not os.path.exists(filepath):
        return False
    try:
        with Dataset(filepath, 'r') as nc_file:
            # 尝试读取文件的维度或变量，确保文件结构正常
            if len(nc_file.dimensions) > 0:
                return True
    except Exception as e:
        print(f"文件检查失败：{filepath}，错误信息：{e}")
    return False

def download_era5_daily(year, month, vvvv):
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": vvvv,
        "year": year,
        "month": month,
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "grid": '0.5/0.5',
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    # 生成文件名
    filename = f"ERA5_hourly_single.0.5x0.5.{vvvv}.{int(year)}{month}.nc"
    #filename = f"ERA5_daily_pressure.1.0x1.0.{int(year)}{month}{day}.nc"

    # 检查文件是否已经存在且完整
    if is_file_complete(filename):
        print(f"文件已存在且完整，跳过下载：{filename}")
        return

    # 如果文件不存在或不完整，则下载
    client = cdsapi.Client()
    try:
        print(f"开始下载文件：{filename}")
        #client.retrieve(dataset, request, filename)
        client.retrieve(dataset, request, filename)
        print(f"文件下载完成：{filename}")
    except Exception as e:
        print(f"下载失败：{filename}，错误信息：{e}")

month_list = [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ]

variables = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure"
    ]

for i in range(2024, 2026):
    for j in month_list:
        for v in variables:
            download_era5_daily(str(int(i)), j, v)