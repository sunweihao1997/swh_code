import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
            'mean_sea_level_pressure', 'sea_surface_temperature', 'skin_temperature',
            'snow_albedo', 'snow_depth', 'snowfall',
            'soil_temperature_level_1', 'soil_type', 'surface_pressure',
            'total_precipitation','soil_temperature_level_2',
            'soil_temperature_level_3','soil_temperature_level_4',
           'volumetric_soil_water_layer_1','volumetric_soil_water_layer_2',
           'volumetric_soil_water_layer_3', 'volumetric_soil_water_layer_4',
            ],
    "year": ["2024"],
    "month": ["05"],
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
        "00:00", "06:00", "12:00",
        "18:00"
    ],
    "data_format": "grib",
    "download_format": "unarchived"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
