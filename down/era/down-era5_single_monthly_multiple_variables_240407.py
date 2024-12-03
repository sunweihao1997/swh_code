import cdsapi


def down_month(yyyy):
    c = cdsapi.Client()

    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'format': 'netcdf',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature',
                '2m_temperature', 'mean_sea_level_pressure', 'skin_temperature',
                'surface_latent_heat_flux', 'surface_pressure', 'surface_sensible_heat_flux',
                'top_net_solar_radiation', 'top_net_thermal_radiation', 'total_precipitation',
            ],
            'year': str(yyyy),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'product_type': 'monthly_averaged_reanalysis',
            'time': '00:00',
        },
        str(yyyy) + '_single_month.nc')

for yy in range(1940, 2024):
    down_month(yy)