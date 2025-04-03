# This code is to download the single level monthly data for 1980-2025



import cdsapi

def download_era5(year):
    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "2m_dewpoint_temperature",
            "2m_temperature",
            "mean_sea_level_pressure",
            "sea_surface_temperature",
            "surface_pressure",
            "total_precipitation",
            "surface_latent_heat_flux",
            "surface_net_solar_radiation",
            "surface_net_solar_radiation_clear_sky",
            "surface_net_thermal_radiation",
            "surface_net_thermal_radiation_clear_sky",
            "surface_sensible_heat_flux",
            "surface_solar_radiation_downward_clear_sky",
            "surface_solar_radiation_downwards",
            "top_net_solar_radiation",
            "top_net_solar_radiation_clear_sky",
            "top_net_thermal_radiation",
            "top_net_thermal_radiation_clear_sky"
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
    client.retrieve(dataset, request, "ERA5_monnthly_single.0.5x0.5." + str(int(year)) + ".nc")

for i in range(2019, 2026):
    download_era5(i)
