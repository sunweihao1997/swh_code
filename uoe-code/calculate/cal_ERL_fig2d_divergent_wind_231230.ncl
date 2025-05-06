; 2023-12-30
; This script calculate the divergent wind
begin

    data = addfile("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_150_ensemble_mean_UVWZ_DIV_and_ttest_1945_1960.nc", "r")

    btal_u       = data->btal_u
    btal_v       = data->btal_v

    btalneu_u    = data->btalneu_u
    btalneu_v    = data->btalneu_v

    ;print(u&lat)


    btal_u@_FillValue = 1.e20
    btal_v@_FillValue = 1.e20
    btalneu_u@_FillValue = 1.e20
    btalneu_v@_FillValue = 1.e20
    replace_ieeenan(btal_u, btal_u@_FillValue, 0)
    replace_ieeenan(btal_v, btal_v@_FillValue, 0)
    replace_ieeenan(btalneu_u, btalneu_u@_FillValue, 0)
    replace_ieeenan(btalneu_v, btalneu_v@_FillValue, 0)

    btal_div     = uv2dv_cfd(btal_u,    btal_v,    btalneu_u&lat, btalneu_u&lon, 2)
    btalneu_div  = uv2dv_cfd(btalneu_u, btalneu_v, btalneu_v&lat, btalneu_v&lon, 2)

;
    btal_ud       = new ( dimsizes(btal_u), typeof(btal_u), btal_u@_FillValue )
    btal_vd       = new ( dimsizes(btal_v), typeof(btal_v), btal_v@_FillValue )
    btalneu_ud    = new ( dimsizes(btalneu_u), typeof(btalneu_u), btalneu_u@_FillValue )
    btalneu_vd    = new ( dimsizes(btalneu_v), typeof(btalneu_v), btalneu_v@_FillValue )


    dv2uvg(btal_div,    btal_ud,    btal_vd)
    dv2uvg(btalneu_div, btalneu_ud, btalneu_vd)
    ;print(ud(10, :, 10))
    copy_VarCoords(btal_u, btal_div )
    copy_VarCoords(btal_u, btalneu_div )
    copy_VarCoords(btal_u, btal_ud )
    copy_VarCoords(btal_u, btal_vd )
    copy_VarCoords(btal_u, btalneu_ud )
    copy_VarCoords(btal_u, btalneu_vd )

    fout = addfile("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/Aerosol_Research_CESM_divergent_wind_divergence_BTAL_BTALnEU_JJAS.nc", "c")
    filedimdef(fout,"time",-1,True)

    fout->btal_div    = btal_div
    fout->btalneu_div = btalneu_div
    fout->btal_ud     = btal_ud
    fout->btal_vd     = btal_vd
    fout->btalneu_ud  = btalneu_ud
    fout->btalneu_vd  = btalneu_vd


end