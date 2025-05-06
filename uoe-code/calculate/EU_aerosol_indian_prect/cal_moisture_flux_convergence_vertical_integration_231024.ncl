; 2023-10-24
; This script is to calculate moisture flux convergence and then calculate vertical integration
begin

    f0 = addfile("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_BTALnEU_Moisture_transportation_ensemble_mean_202310.nc", "r")
    f1 = addfile("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTAL_PS_ensemble_mean_231020.nc", "r")
    f2 = addfile("/exports/csce/datastore/geos/users/s2618078/data/analysis_data/analysis_EU_aerosol_climate_effect/BTALnEU_PS_ensemble_mean_231020.nc", "r")
   

    ;print(f0)
    mt_u_BTAL = f0->transport_x_BTAL
    mt_v_BTAL = f0->transport_y_BTAL
    mt_u_BTALnEU = f0->transport_x_BTALnEU
    mt_v_BTALnEU = f0->transport_y_BTALnEU

    psfc1     = f1->PS
    psfc2     = f2->PS

    lon       = f0->lon
    lat       = f0->lat
    lev       = f0->lev

    ;print(lev(:20))

    if (any(isnan_ieee(mt_u_BTAL))) then
        value = 1.e20
        replace_ieeenan(mt_u_BTAL, value, 0)
        replace_ieeenan(mt_v_BTAL, value, 0)
        replace_ieeenan(mt_u_BTALnEU, value, 0)
        replace_ieeenan(mt_v_BTALnEU, value, 0)
        mt_u_BTAL@_FillValue = value
        mt_v_BTAL@_FillValue = value
        mt_u_BTALnEU@_FillValue = value
        mt_u_BTALnEU@_FillValue = value
    end if

    ; Calculate the divergence
    mt_dv_BTAL     =  uv2dv_cfd(mt_u_BTAL, mt_v_BTAL, lat, lon, 2)
    mt_dv_BTALnEU  =  uv2dv_cfd(mt_u_BTALnEU, mt_v_BTALnEU, lat, lon, 2)

    ; copy variable coordinate
    copy_VarCoords(mt_u_BTAL, mt_dv_BTAL)
    copy_VarCoords(mt_u_BTAL, mt_dv_BTALnEU)

    print("Success to calculate convergence")

    ;printVarSummary(mt_dv_BTAL)
    pbot   = 1000
    ptop   = 300
    mt_vint_BTAL     =  new((/1883,96,144/),double)
    mt_vint_BTALnEU  =  new((/1883,96,144/),double)
    do tt = 0, 1882
        print(tt)
        do latt = 0, 95
            do lonn = 0, 143
                mt_vint_BTAL(tt, latt, lonn)    = vibeta(lev, mt_dv_BTAL(tt, :, latt, lonn), 1, psfc1(tt, latt, lonn), pbot, ptop)
                mt_vint_BTALnEU(tt, latt, lonn) = vibeta(lev, mt_dv_BTALnEU(tt, :, latt, lonn), 1, psfc2(tt, latt, lonn), pbot, ptop)
            end do
        end do
    end do
    ;mt_vint_BTAL     =  vibeta(lev, mt_dv_BTAL, 1, psfc1, pbot, ptop)
    ;mt_vint_BTALnEU  =  vibeta(lev, mt_dv_BTALnEU, 1, psfc2, pbot, ptop)
    ; =========== 4. Write to ncfile==================
    out_path      =  "/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/"
    out_file      =  "BTAL_moisture_flux_integration_1850_150years.nc"

    fout_1        =  addfile(out_path + out_file, "c")

    ;filedimdef(fout_1,"time",-1,True)

    fout_1->mt_vint_BTAL     =  mt_vint_BTAL
    fout_1->mt_vint_BTALnEU  =  mt_vint_BTALnEU

    
end