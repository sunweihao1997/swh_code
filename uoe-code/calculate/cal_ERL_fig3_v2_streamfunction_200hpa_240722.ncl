; 2024-7-22
; This script calculate the streamfunction at 200 hPa
begin

    data = addfile("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/EUI_CESM_BTAL_BTALnEU_high_level_but_single_level_200_ensemble_mean_UVWZ_DIV_and_ttest.nc", "r")

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

;
    btal_sf     = new ( dimsizes(btal_u), typeof(btal_u), btal_u@_FillValue )
    btal_vp     = new ( dimsizes(btal_v), typeof(btal_v), btal_v@_FillValue )
    btalneu_sf  = new ( dimsizes(btal_u), typeof(btal_u), btal_u@_FillValue )
    btalneu_vp  = new ( dimsizes(btal_v), typeof(btal_v), btal_v@_FillValue )


    uv2sfvpf(btal_u,    btal_v,    btal_sf,    btal_vp)
    uv2sfvpf(btalneu_u, btalneu_v, btalneu_sf, btalneu_vp)
    ;print(ud(10, :, 10))
    copy_VarCoords(btal_u, btal_sf )
    copy_VarCoords(btal_u, btal_vp )
    copy_VarCoords(btal_u, btalneu_sf )
    copy_VarCoords(btal_u, btalneu_vp )

    fout = addfile("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/Aerosol_Research_CESM_BTAL_BTALnEU_200hPa_streamfunction_velocity_potential.nc", "c")
    filedimdef(fout,"time",-1,True)

    fout->btal_sf        = btal_sf
    fout->btal_vp        = btal_vp
    fout->btalneu_sf     = btalneu_sf
    fout->btalneu_vp     = btalneu_vp

end