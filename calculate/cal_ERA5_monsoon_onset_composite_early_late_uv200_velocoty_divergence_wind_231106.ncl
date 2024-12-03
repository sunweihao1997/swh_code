; 2023-11-05
; This script deal with the ERA5 abnormal years' composite result. The goal is to see the characteristics of the upper-level convective intensity
; Main purpose: volecity potential and divergent wind
begin
    data_path = "/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/"

    ufile     = addfile(data_path + "u_component_of_wind_composite_abnormal.nc", "r")
    vfile     = addfile(data_path + "v_component_of_wind_composite_abnormal.nc", "r")

    lev       = 200
    ;print(ufile)
    ; Read-in data arrray
    u_early   = ufile->u_early(:, {200}, :, :)
    u_late    = ufile->u_late(:, {200}, :, :)
    v_early   = vfile->v_early(:, {200}, :, :)
    v_late    = vfile->v_late(:, {200}, :, :)
    lat       = ufile->lat
    lon       = ufile->lon

    ;printVarSummary(u_early)
    ; if (any(isnan_ieee(u_early))) then
         value = 1.e20
         replace_ieeenan(u_early, value, 0)
         replace_ieeenan(u_late,  value, 0)
         replace_ieeenan(v_early, value, 0)
         replace_ieeenan(v_late,  value, 0)
         u_early@_FillValue = value
         v_early@_FillValue = value
         u_late@_FillValue  = value
         v_late@_FillValue  = value
    ; end if
    ; printVarSummary(u_early) 
    ;printVarSummary(u_early)


    boundOpt  = 2 ; Boundary points are estimated using one-sided difference schemes normal to the boundary.
    dv_early  = uv2dv_cfd(u_early, v_early, lat, lon, boundOpt)
    dv_late   = uv2dv_cfd(u_late,  v_late,  lat, lon, boundOpt)
    ;printVarSummary(dv_early)
    ;printMinMax(dv_early, 0)

    ;printVarSummary(dv_early)
    copy_VarCoords(u_early, dv_early)
    copy_VarCoords(v_early, dv_late)

    ; =========== Calculate the divergent wind ==================
    dvwind_early  =  dv2uvF(dv_early)
    dvwind_late   =  dv2uvF(dv_late)
    ;printVarSummary(dvwind_early)
    dv_u_early    =  dvwind_early(0, :, :, :)
    dv_v_early    =  dvwind_early(1, :, :, :)
    dv_u_late     =  dvwind_late(0,  :, :, :)
    dv_v_late     =  dvwind_late(1,  :, :, :)
    printMinMax(dvwind_early, 0)
    copy_VarCoords(u_early, dv_u_early)
    copy_VarCoords(u_early, dv_v_early)
    copy_VarCoords(u_early, dv_u_late)
    copy_VarCoords(u_early, dv_v_late)

    ; ============ Write to ncfile ===============================
    ;system("rm /home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/composite_ERA5_abnormal_years_divergent_wind_and_divergence_200.nc")
    ;fout = addfile("/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/composite_ERA5_abnormal_years_divergent_wind_late.nc","c")
    setfileoption("nc","Format","LargeFile")
    fout = addfile("/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/composite_ERA5_abnormal_years_divergent_wind_and_divergence_200.nc","c")

    ;filedimdef(fout,"time",-1,True)
    fout->dv_early    =  (/dv_early/)
    fout->dv_late     =  (/dv_late/)
    fout->dv_u_early  =  (/dv_u_early/)
    fout->dv_v_early  =  (/dv_v_early/)
    fout->dv_u_late   =  (/dv_u_late/)
    fout->dv_v_late   =  (/dv_v_late/)

    fileatt  =  True
    ;fileatt@time  =  "2023-11-5"
    fileatt@description  =  "This file save the divergence and divergent wind using ERA5 composite early/late years U/V data, script name is cal_ERA5_monsoon_onset_composite_early_late_uv200_velocoty_divergence_wind_231105.ncl"  
end