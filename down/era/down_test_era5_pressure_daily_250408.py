import cdsapi


def download_era5_daily(year, month, day):
    dataset = "derived-era5-pressure-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": [
            "fraction_of_cloud_cover",
        ],
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
    }

    client = cdsapi.Client()
    #client.retrieve(dataset, request, "ERA5_daily_pressure.1.0x1.0." + str(int(year)) + month+day+".nc")
    client.retrieve(dataset, request, "namefile")

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

#for i in range(1980, 2026):
#    for j in month_list:
#        for k in day_list:
#            download_era5_daily(i, j, k)
download_era5_daily(str(1980), "02", "29")