;2025-5-7
;Use CESM simulation data to calculate the stream function at 850 hPa
begin
f1 = addfile("/home/sun/data/download_data/data/model_data/ensemble_JJA_corrected/CESM_BTAL_BTALnEU_JJA_UV_ensemble_average_850hpa.nc","r")

u_btal = f1->JJA_U_average_BTAL
v_btal = f1->JJA_V_average_BTAL
u_btalneu = f1->JJA_U_average_BTALnEU
v_btalneu = f1->JJA_V_average_BTALnEU


; Delete NAN value
if (any(isnan_ieee(u_btal))) then
    value = 1.e20
    replace_ieeenan(u_btal, value, 0)
    replace_ieeenan(v_btal, value, 0)
    replace_ieeenan(u_btalneu, value, 0)
    replace_ieeenan(v_btalneu, value, 0)
    u_btal@_FillValue = value
    v_btal@_FillValue = value
    u_btalneu@_FillValue = value
    v_btalneu@_FillValue = value
end if

btal_sf     = new ( dimsizes(u_btal), typeof(u_btal), u_btal@_FillValue )
btal_vp     = new ( dimsizes(u_btal), typeof(u_btal), u_btal@_FillValue )
btalneu_sf  = new ( dimsizes(u_btal), typeof(u_btal), u_btal@_FillValue )
btalneu_vp  = new ( dimsizes(u_btal), typeof(u_btal), u_btal@_FillValue )

uv2sfvpf(u_btal, v_btal,       btal_sf,    btal_vp)
uv2sfvpf(u_btalneu, v_btalneu, btalneu_sf, btalneu_vp)

copy_VarCoords(u_btal, btal_sf )
copy_VarCoords(u_btal, btal_vp )
copy_VarCoords(u_btal, btalneu_sf )
copy_VarCoords(u_btal, btalneu_vp )

fout = addfile("/home/sun/data/download_data/data/analysis_data/analysis_EU_aerosol_climate_effect/Aerosol_Research_CESM_BTAL_BTALnEU_850hPa_streamfunction_velocity_potential.nc", "c")
filedimdef(fout,"time",-1,True)

fout->btal_sf        = btal_sf
fout->btal_vp        = btal_vp
fout->btalneu_sf     = btalneu_sf
fout->btalneu_vp     = btalneu_vp

end