import cdsapi

c = cdsapi.Client()

def down_single_year(yyyy, mm):
    dataset = "reanalysis-era5-single-levels"
    request = {
        "product_type": ["reanalysis"],
        "variable": [
            "significant_height_of_combined_wind_waves_and_swell",
            "significant_height_of_total_swell",
            "significant_height_of_wind_waves"
        ],
        "year": [str(yyyy)],
        "month": mm,
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
        "grid": '1.0/1.0',
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request, "ERA5_wave.0.5x0.5." + str(int(yyyy)) + ".nc")

start_yy = 2000 ; end_yy = 2025

months = [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ]

for yyy in range(start_yy, end_yy):
    for mm in months:
        down_single_year(yyy, mm)
