import cdsapi

c = cdsapi.Client()

def down_mon(yyyy):
    c.retrieve(
        'reanalysis-era5-pressure-levels-monthly-means',
        {
            'format': 'netcdf',
            'product_type': 'monthly_averaged_reanalysis',
            'variable': [
                'geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind', 'vertical_velocity',
            ],
            'pressure_level': [
                '10', '20', '30',
                '50', '70', '100',
                '125', '150', '175',
                '200', '225', '250',
                '300', '350', '400',
                '450', '500', '550',
                '600', '650', '700',
                '750', '775', '800',
                '825', '850', '875',
                '900', '925', '950',
                '975', '1000',
            ],
            'year': str(yyyy),
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'time': '00:00',
        },
        'ERA5_{}_monthly_pressure_UTVWZSH.nc'.format(yyyy))

for i in range(1940, 2023):
    down_mon(i)