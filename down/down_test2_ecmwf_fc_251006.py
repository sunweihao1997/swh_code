'''
2025-10-6
This script test ecmwf-opendata to download forecast data
'''
from ecmwf.opendata import Client

client = Client(source="ecmwf")

client.download(
    param="msl",
    type="fc",
    step=24,
    target="data.grib2",
)
