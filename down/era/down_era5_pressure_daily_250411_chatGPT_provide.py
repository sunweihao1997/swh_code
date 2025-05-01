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

def download_era5_daily(year, month, day, vvvv):
    dataset = "derived-era5-pressure-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": vvvv,
        "year": year,
        "month": month,
        "day": day,
        "pressure_level": [
                "10","50", 
                "100", "125", "150",
                "175", "200", "225",
                "250", "300", "350",
                "400", "450", "500",
                "550", "600", "650",
                "700", "750", "775",
                "800", "825", "850",
                "875", "900", "925",
                "950", "975", "1000"
            ],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "3_hourly",
        "grid": '1.0/1.0',
        #"data_format": "netcdf",
        "download_format": "unarchived"
    }

    # 生成文件名
    filename = f"ERA5_daily_pressure.1.0x1.0.{vvvv}.{int(year)}{month}{day}.nc"
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
day_list    = [
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
        ]

variables = [
            "fraction_of_cloud_cover",
            "geopotential",
            "relative_humidity",
            "specific_cloud_ice_water_content",
            "specific_cloud_liquid_water_content",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity",
        ]

for i in range(1980, 2026):
    for j in month_list:
        for k in day_list:
            for v in variables:
                download_era5_daily(str(int(i)), j, k, v)