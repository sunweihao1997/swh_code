import cdsapi

def download_function(yyyy, mmmm, dddd):
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure"
    ],
        "year": [
            str(int(yyyy))
        ],
        "month": mmmm,
        "day":dddd,
        "grid": '0.5/0.5',
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
        "data_format": "netcdf",
        "download_format": "unarchived"
    }
    
    client = cdsapi.Client()
    client.retrieve(dataset, request, f"ERA5_single_ship_compare.{int(yyyy)}{mmmm}{dddd}.nc")

month_list = [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ]

day_list =  [
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

for i in range(2024, 2026):
    for j in month_list:
        for k in day_list:
            download_function(i, j, k)