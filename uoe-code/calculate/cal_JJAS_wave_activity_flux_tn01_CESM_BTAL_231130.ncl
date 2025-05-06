;2023-11-30
;This script is to calculate TN wave activity flux

begin

    path_src = "/exports/csce/datastore/geos/users/s2618078/data/model_data/ensemble_JJAS/"

    ; 1. monthly-mean data hgt units:m
    f0       = addfile(path_src + "CESM_BTAL_JJAS_Z3_ensemble.nc", "r")

    ; 2. monthly climatology data
    ; 2.1 hgt could use f0

    ; 2.2 air temperature units: degC (need to be change)
    f_air    = addfile(path_src + "CESM_BTAL_JJAS_T_ensemble.nc", "r")

    ; 2.3 zonal wind m/s
    f_u      = addfile(path_src + "CESM_BTAL_JJAS_U_ensemble.nc", "r")

    ; 2.4 meridional wind m/s
    f_v      = addfile(path_src + "CESM_BTAL_JJAS_V_ensemble.nc", "r")

    ; dimension information
    time = f0->time
    lat  = f0->lat
    lon  = f0->lon
    level= f0->lev

    ; ----------------------

    ntime = dimsizes(time)
    nlat  = dimsizes(lat)
    nlon  = dimsizes(lon)
    nlevel= dimsizes(level)

    ; 3. Data pre-process
    ; 3.1 Calculate ensemble-average for hgt
    hgt_1   = f0->JJAS_Z3_1
    hgt_2   = f0->JJAS_Z3_2
    hgt_3   = f0->JJAS_Z3_3
    hgt_4   = f0->JJAS_Z3_4
    hgt_5   = f0->JJAS_Z3_5
    hgt_6   = f0->JJAS_Z3_6
    hgt_7   = f0->JJAS_Z3_7
    hgt_8   = f0->JJAS_Z3_8

    ; 3.1.1 Replace the nan data
    if (any(isnan_ieee(hgt_1))) then
        value = 1.e20
        replace_ieeenan(hgt_1, value, 0)
        replace_ieeenan(hgt_2, value, 0)
        replace_ieeenan(hgt_3, value, 0)
        replace_ieeenan(hgt_4, value, 0)
        replace_ieeenan(hgt_5, value, 0)
        replace_ieeenan(hgt_6, value, 0)
        replace_ieeenan(hgt_7, value, 0)
        replace_ieeenan(hgt_8, value, 0)
    end if 

    hgt_1@_FillValue = value
    hgt_2@_FillValue = value
    hgt_3@_FillValue = value
    hgt_4@_FillValue = value
    hgt_5@_FillValue = value
    hgt_6@_FillValue = value
    hgt_7@_FillValue = value
    hgt_8@_FillValue = value

    hgt_1_climatology = dim_avg_n_Wrap(hgt_1, 0)
    hgt_2_climatology = dim_avg_n_Wrap(hgt_2, 0)
    hgt_3_climatology = dim_avg_n_Wrap(hgt_3, 0)
    hgt_4_climatology = dim_avg_n_Wrap(hgt_4, 0)
    hgt_5_climatology = dim_avg_n_Wrap(hgt_5, 0)
    hgt_6_climatology = dim_avg_n_Wrap(hgt_6, 0)
    hgt_7_climatology = dim_avg_n_Wrap(hgt_7, 0)
    hgt_8_climatology = dim_avg_n_Wrap(hgt_8, 0)

    printVarSummary(hgt_1_climatology)

    hgt_ensemble_climatology = hgt_1_climatology
    hgt_ensemble_climatology = (hgt_1_climatology + hgt_2_climatology + hgt_3_climatology + hgt_4_climatology + hgt_5_climatology + hgt_6_climatology + hgt_7_climatology + hgt_8_climatology) / 8

    printVarSummary(hgt_ensemble_climatology)


    
end
