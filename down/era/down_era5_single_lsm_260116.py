import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["land_sea_mask"],
    "year": ["2023"],
    "month": ["09"],
    "day": ["28"],
    "time": ["21:00"],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "grid": '0.5/0.5',
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()
