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

   
    copy_VarCoords(u_ssp(0,:,:,:),div_ssp)
    printVarSummary(div_ssp);[lev | 29] x [lat | 192] x [lon | 288]
    copy_VarCoords(u_ntcf(0,:,:,:),div_ntcf)
    printVarSummary(div_ntcf);[lev | 29] x [lat | 192] x [lon | 288]
    
    printVarSummary(ps_ssp); [lat | 192] x [lon | 288]
 
;--------- vibeta for div_ssp  and div_ntcf---------------
    pbot    = 1100.
    ptop    = 10
    linlog  =  1
    ; for ssp
    psfc    =  ps_ssp ; originally 1013.
    x       =  div_ssp
    p       =  div_ssp&lev ;int64
    ptmp    =  tofloat(p)  ;float
    copy_VarCoords(p,ptmp)
    ;for ntcf
    psfc_ntcf     =  ps_ntcf      ; originally 1013.
    x_ntcf        =  div_ntcf 
    p_ntcf        =  div_ntcf&lev ;int64
    ptmp_ntcf     =  tofloat(p_ntcf)  ;float
    copy_VarCoords(p_ntcf ,ptmp_ntcf)
    
    vint_ssp  = vibeta (ptmp,x(lat|:,lon|:,lev|:),linlog,psfc,pbot,ptop)  ;min=-1.04132e-05   max=8.76435e-06
    vint_ntcf = vibeta (ptmp_ntcf,x_ntcf(lat|:,lon|:,lev|:),linlog,psfc,pbot,ptop);min=-7.84776e-06   max=7.90231e-06
    printMinMax(vint_ntcf,False)
    printMinMax(vint_ssp,False)
; --------------- Write to ncfile ---------------------------------
    ;system("rm -rf /home/sun/data/AerChemMIP/process/modelmean_climate_water_transportation_integral_MJJAS_UKESM1-0-LL.nc")
    fout = addfile("/home/sun/data/model_data/climate/b1850_exp/b1850_control_Indian_climate_water_transportation_integral_daily_liuxl.nc", "c")
    filedimdef(fout,"time",-1,True)
;
    fout->vert_integral_control  = vint_ssp
    fout->vert_integral_indian   = vint_ntcf

;
    ;; Set file attribute
    setfileoption(fout,"DefineMode",True)
    fAtt = True
    fAtt@description = "Create on 4/Jul/2024. This file is water transportation vertical integral"
    fileattdef(fout,fAtt)
    end