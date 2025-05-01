import cdsapi

dataset = "sis-water-level-change-indicators-cmip6"
request = {
    "variable": [
        "surge_level",
        "total_water_level",
        "mean_sea_level",
        "annual_mean_of_highest_high_water"
    ],
    "derived_variable": [
        "absolute_change",
        "absolute_value",
        "percentage_change"
    ],
    "product_type": ["multi_model_ensemble"],
    "multi_model_ensemble_statistic": ["ensemble_mean"],
    "statistic": ["100_year"],
    "confidence_interval": ["best_fit"],
    "experiment": [
        "historical",
        "future"
    ],
    "period": [
        "1951_1980",
        "1985_2014",
        "2021_2050"
    ]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()