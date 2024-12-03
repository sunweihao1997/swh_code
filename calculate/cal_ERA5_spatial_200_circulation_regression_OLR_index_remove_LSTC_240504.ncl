;;;2024-4-5
;;;This script calculate the regression of wind to the LSTC (OLR removed)
;;; Spatial circulation 200 hPa
begin
    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================
    ; Wind data
    fu = addfile("/home/sun/data/download/part_ERA5_pressure_level/u_FMA_1980-2020.nc", "r")
    fv = addfile("/home/sun/data/download/part_ERA5_pressure_level/v_FMA_1980-2020.nc", "r")
    fw = addfile("/home/sun/data/download/part_ERA5_pressure_level/w_FMA_1980-2020.nc", "r")

    ; Land-Sea thermal contrast
    f1 = addfile("/home/sun/data/ERA5_data_monsoon_onset/regression/regression_uv_to_OLR_to_LSTC_remove_each_other_FMA_avg.nc", "r")

    ; Read data
    u  = fu->u(:, {850}, :, :)
    v  = fv->v(:, {850}, :, :)
    w  = fw->w(:, {850}, :, :)

    olr = f1->lstc_remove_olr

    olr = dim_standardize_Wrap(olr, 1)
;    print(olr)

;    ; Calculates and removes the mean of the given dimension(s) at all other dimensions.
    olr_abnormal  = dim_rmvmean_n_Wrap(olr, 0)
    u_abnormal    = dim_rmvmean_n_Wrap(u, 0)
    v_abnormal    = dim_rmvmean_n_Wrap(v, 0)
    w_abnormal    = dim_rmvmean_n_Wrap(w, 0)
;
;    ;print(u_abnormal)
;    ; Detrend
    olr_detrend  = olr_abnormal
    olr_detrend  = (/dtrend_n(olr_abnormal, True, 0)/)

    u_detrend     = u_abnormal
    u_detrend     = (/dtrend_n(u_abnormal, True, 0)/)
    v_detrend     = v_abnormal
    v_detrend     = (/dtrend_n(v_abnormal, True, 0)/)
    w_detrend     = w_abnormal
    w_detrend     = (/dtrend_n(w_abnormal, True, 0)/)
;    printVarSummary(u_detrend)
;
;
;    ; Change the dimension sequence
;    ;printVarSummary(u_abnormal)
    u_detrend_    = u_detrend
    v_detrend_    = v_detrend
    w_detrend_    = w_detrend

;    ;printVarSummary(lstc_detrend)
;    ;printVarSummary(u_detrend_)
;
;    ; Delete NAN value
    if (any(isnan_ieee(u_detrend_))) then
        value = 1.e20
        replace_ieeenan(u_detrend_, value, 0)
        replace_ieeenan(v_detrend_, value, 0)
        replace_ieeenan(w_detrend_, value, 0)
        w_detrend_@_FillValue = value
        u_detrend_@_FillValue = value
        v_detrend_@_FillValue = value
    end if
    ;printVarSummary(w_detrend_)
    value = 1.e20
    replace_ieeenan(u_detrend, value, 0)
;
;    ; ======Add attributes and write to nc file======
;
;    lstc_detrend_april@units = "W m-2"
;    ;lstc_detrend_april!0     = "year"
;    ;lstc_detrend_april&year  = lstc_detrend&year
;
;    fout = addfile("/home/sun/data/ERA5_data_monsoon_onset/index/OLR_index_April_ERA5_1959to2021.nc", "c")
;    filedimdef(fout,"time",-1,True)
;
;    fout->LSTC_april = lstc_detrend_april
;
;    ; Set file attribute
;    setfileoption(fout,"DefineMode",True)
;    fAtt = True
;    fAtt@description = "Create on 05/5/2023. This is the April mean OLR index from 1959 to 2021."
;    fileattdef(fout,fAtt)
;
;    ;=================================================
;    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================
;
;    ; ================ Data processing2: calculate regression =======================
;    ;printVarSummary(u_detrend)
;    ;printVarSummary(lstc_detrend_april)
;    printVarSummary(u_detrend_)
    rc_u = regCoef_n(olr_detrend*-1, u_detrend_, 0, 0)  
    rc_v = regCoef_n(olr_detrend*-1, v_detrend_, 0, 0)  
    rc_w = regCoef_n(olr_detrend*-1, w_detrend_, 0, 0)  
    ;printVarSummary(rc_u)

    ;printVarSummary(u_detrend_)

;    printVarSummary(rc_u)

    ; ================ Write to the nc file ==========================================

    copy_VarCoords(u(0, :, :), rc_u)
    copy_VarCoords(v(0, :, :), rc_v)
    copy_VarCoords(w(0, :, :), rc_w)

    ;printVarSummary(u(0, :, :, {80}))

    ; delete irrelevant attributes
    delete(rc_u@nptxy)
    delete(rc_u@rstd)
    delete(rc_u@yintercept)
    delete(rc_u@tval)
    delete(rc_v@nptxy)
    delete(rc_v@rstd)
    delete(rc_v@yintercept)
    delete(rc_v@tval)
    delete(rc_w@nptxy)
    delete(rc_w@rstd)
    delete(rc_w@yintercept)
    delete(rc_w@tval)

;    printMinMax(lstc_detrend, 0)
;    printMinMax(rc_u, 0)
    system("rm /home/sun/data/ERA5_data_monsoon_onset/regression/ERA5_regression_LSTC-remove-OLR_850_spatial_circulation.nc")
    fout = addfile("/home/sun/data/ERA5_data_monsoon_onset/regression/ERA5_regression_LSTC-remove-OLR_850_spatial_circulation.nc", "c")
    filedimdef(fout,"time",-1,True)
;
    fout->rc_u = rc_u
    fout->rc_v = rc_v
    fout->rc_w = rc_w

    

    ;;; Set file attribute
    setfileoption(fout,"DefineMode",True)
    fAtt = True
    fAtt@description = "Create on 6/May/2024. This file include regression from u/v/w to the LSTC index at 850 hPa spatial pattern."
    fileattdef(fout,fAtt)
end
