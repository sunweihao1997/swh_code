; 2024-5-28
; This script is to use the ncl function to calculate the vertical integral of the moisture transportation
begin

    f0 = addfile("/home/sun/data/model_data/climate/b1850_exp/b1850_control_climate_atmosphere.nc", "r")
    f1 = addfile("/home/sun/data/model_data/climate/b1850_exp/b1850_indian_climate_atmosphere3.nc", "r")

    u_ssp    = f0->U
    v_ssp    = f0->V
    hus_ssp  = f0->Q
    ps_ssp0   = f0->PS

    u_ntcf    = f1->U
    v_ntcf    = f1->V
    hus_ntcf  = f1->Q
    ps_ntcf0   = f1->PS

    lon       = f0->lon
    lat       = f0->lat
    lev       = f0->lev
    lev       = lev*100

    ;print(lev(:20))
    ; remove the NaN with FillValue
    if (any(isnan_ieee(u_ssp))) then
        print("Include NAN!")
        value = 1.e20
        replace_ieeenan(u_ssp   , value, 0)
        replace_ieeenan(v_ssp   , value, 0)
        replace_ieeenan(hus_ssp , value, 0)
        replace_ieeenan(ps_ssp0  , value, 0)
        replace_ieeenan(u_ntcf  , value, 0)
        replace_ieeenan(v_ntcf  , value, 0)
        replace_ieeenan(hus_ntcf, value, 0)
        replace_ieeenan(ps_ntcf0 , value, 0)
        u_ssp@_FillValue      = value
        v_ssp@_FillValue      = value
        hus_ssp@_FillValue    = value
        ps_ssp0@_FillValue     = value
        u_ntcf@_FillValue     = value
        v_ntcf@_FillValue     = value
        hus_ntcf@_FillValue   = value
        ps_ntcf0@_FillValue    = value
    end if

    ;printMinMax(ps_ssp, 0)
    ;print(lev)
    ; --------- Calculate the moisture transportation on single level ---------------
    uq_ssp  = u_ssp * hus_ssp / 9.8
    vq_ssp  = v_ssp * hus_ssp / 9.8
    uq_ntcf = u_ntcf * hus_ntcf / 9.8
    vq_ntcf = v_ntcf * hus_ntcf / 9.8

    div_ssp0  = uv2dv_cfd(uq_ssp,  vq_ssp,  lat, lon, 2)
    div_ntcf0 = uv2dv_cfd(uq_ntcf, vq_ntcf, lat, lon, 2)

    div_ssp   = dim_avg_n_Wrap(div_ssp0(120:150, :, :, :), 0)
    div_ntcf  = dim_avg_n_Wrap(div_ntcf0(120:150, :, :, :), 0)

    ps_ssp    = dim_avg_n_Wrap(ps_ssp0(120:150, :, :),  0)
    ps_ntcf   = dim_avg_n_Wrap(ps_ntcf0(120:150, :, :), 0)

    ;printMinMax(div_ntcf, 0)

    ; -------- Calculate the vertical integral ---------
    vert_integral_ssp  = new((/192, 288/), double)
    vert_integral_ntcf = new((/192, 288/), double)

    pbot   = 100000
    ptop   = 30000
    do latt = 0, 191
        do lonn = 0, 287
            if ismissing(div_ssp(15, latt, lonn)) then 
                ;print("Continue")
                vert_integral_ssp(latt, lonn)  = 0
                vert_integral_ntcf(latt, lonn) = 0
            else if ismissing(ps_ssp(latt, lonn)) then
                vert_integral_ssp(latt, lonn)  = 0
                vert_integral_ntcf(latt, lonn) = 0
            else
                ;print(ps_ssp(:, :))
                ;print(div_ssp(:, latt, lonn))
                vert_integral_ssp(latt, lonn)  = vibeta(lev*100, div_ssp(:, latt, lonn),  1, ps_ssp(latt, lonn),  pbot, ptop)
                vert_integral_ntcf(latt, lonn) = vibeta(lev*100, div_ntcf(:, latt, lonn), 1, ps_ntcf(latt, lonn), pbot, ptop)
;            if vert_integral_ssp(latt, lonn) > 1e-5 .or. vert_integral_ssp(latt, lonn) < -1e-5 then 
;                vert_integral_ssp(latt, lonn)  = 0
;                vert_integral_ntcf(latt, lonn) = 0
;            end if
            end if
            end if
        end do
    end do

    printMinMax(vert_integral_ssp, 0)

    copy_VarCoords(ps_ssp, vert_integral_ssp)
    copy_VarCoords(ps_ssp, vert_integral_ntcf)

    ; Write to ncfile
    ;system("rm -rf /home/sun/data/AerChemMIP/process/modelmean_climate_water_transportation_integral_MJJAS_UKESM1-0-LL.nc")
    fout = addfile("/home/sun/data/model_data/climate/b1850_exp/b1850_control_Indian_climate_water_transportation_integral_daily.nc", "c")
    filedimdef(fout,"time",-1,True)
;
    fout->vert_integral_control  = vert_integral_ssp
    fout->vert_integral_indian   = vert_integral_ntcf

;
    ;; Set file attribute
    setfileoption(fout,"DefineMode",True)
    fAtt = True
    fAtt@description = "Create on 4/Jul/2024. This file is water transportation vertical integral"
    fileattdef(fout,fAtt)

    
end