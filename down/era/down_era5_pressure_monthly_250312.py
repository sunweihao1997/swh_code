# This code is to download the pressure level monthly data for 1980-2025

import cdsapi

def download_era5(year):
    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "fraction_of_cloud_cover",
            "geopotential",
            "relative_humidity",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity"
        ],
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
        "year": [str(int(year))],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "grid": '0.5/0.5',
        "data_format": "netcdf",
        "download_format": "unarchived"
    }

    client = cdsapi.Client()
    client.retrieve(dataset, request, "ERA5_monnthly_pressure.0.5x0.5." + str(int(year)) + ".nc")

for i in range(2014, 2026):
    download_era5(i)
