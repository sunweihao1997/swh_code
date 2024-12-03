;2023-5-5
;This script calculate the regression of wind to the OLR
begin
    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================
    ; Wind data
    fpsl = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/monthly/ERA5_1980_2021_monthly_March_April_1000_700_qV.nc", "r")

    ; Land-Sea thermal contrast
    f1 = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_mar.nc", "r")

    ; Read data
    u  = fpsl->Mar_qu
    v  = fpsl->Mar_qv

    lstc_remove        = f1->OLR_mari_Afri
    olr                = f1->LSTC_psl_IOB

    ; Delete NAN value
    if (any(isnan_ieee(u))) then
        value = 1.e20
        replace_ieeenan(u, value, 0)
        replace_ieeenan(v, value, 0)
        replace_ieeenan(olr, value, 0)
        replace_ieeenan(lstc_remove, value, 0)
        u@_FillValue = value
        v@_FillValue = value
        olr@_FillValue = value
        lstc_remove@_FillValue = value

    end if


    ; remove the LSTC signal
    regh1=new(1,"float",0)
    tval1=new(1,"float",0)
    nptxy1=new(1,"integer",0)
    b1            = regcoef(lstc_remove, olr, tval1, nptxy1)
    olr_mean      = dim_avg(olr)
    a1            = olr_mean - b1*dim_avg(lstc_remove)

    olr_remove_lstc = olr - (b1 * lstc_remove + a1)

    

    ; Calculates and removes the mean of the given dimension(s) at all other dimensions.
    lstc_abnormal = dim_rmvmean_n_Wrap(olr_remove_lstc, 0)
    u_abnormal    = dim_rmvmean_n_Wrap(u, 0)
    v_abnormal    = dim_rmvmean_n_Wrap(v, 0)

    ; calculate divergence of moisture transportation
    psl_abnormal  = uv2dv_cfd(u_abnormal, v_abnormal, u&lat, u&lon, 2)
    copy_VarCoords(u, psl_abnormal)

    ;print(u_abnormal)
    ; Detrend
    lstc_detrend  = lstc_abnormal
    lstc_detrend  = (/dim_standardize_n_Wrap(dtrend_n(lstc_abnormal, True, 0), 1, 0)/)

    u_detrend     = u_abnormal
    u_detrend     = (/dtrend_n(u_abnormal, True, 0)/)
    v_detrend     = v_abnormal
    v_detrend     = (/dtrend_n(v_abnormal, True, 0)/)
    p_detrend     = psl_abnormal
    p_detrend     = (/dtrend_n(psl_abnormal, True, 0)/)


    ; Change the dimension sequence
    ;printVarSummary(u_abnormal)
    u_detrend_    = u_detrend
    v_detrend_    = v_detrend
    p_detrend_    = p_detrend
    ;printVarSummary(lstc_detrend)
    ;printVarSummary(u_detrend_)


    ;=================================================
    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================

    ; ================ Data processing2: calculate regression =======================
    ;printVarSummary(u_detrend)
    ;printVarSummary(lstc_detrend_april)
    rc_u = regCoef_n(lstc_detrend, u_detrend_, 0, 0)  
    rc_v = regCoef_n(lstc_detrend, v_detrend_, 0, 0)  
    rc_p = regCoef_n(lstc_detrend, p_detrend_, 0, 0)
    
    ;Calculate the correlation
    r_u_lstc = escorc(lstc_detrend,u_detrend_(lat|:,lon|:,time|:))
    r_v_lstc = escorc(lstc_detrend,v_detrend_(lat|:,lon|:,time|:))
    r_p_lstc = escorc(lstc_detrend,p_detrend_(lat|:,lon|:,time|:))
    copy_VarCoords(u_detrend_(0,:,:),r_u_lstc)
    copy_VarCoords(v_detrend_(0,:,:),r_v_lstc)
    copy_VarCoords(p_detrend_(0,:,:),r_p_lstc)

    Nr       = dimsizes(lstc_detrend)
    t_u      = r_u_lstc*sqrt((Nr-2)/(1-r_u_lstc^2))
    t_v      = r_v_lstc*sqrt((Nr-2)/(1-r_v_lstc^2))
    t_p      = r_p_lstc*sqrt((Nr-2)/(1-r_p_lstc^2))
    ;test significant
    Nr          = dimsizes(lstc_detrend)
    prob_u_lstc = rtest(r_u_lstc , Nr, 0)
    prob_v_lstc = rtest(r_v_lstc , Nr, 0)
    prob_p_lstc = rtest(r_p_lstc , Nr, 0)

    ; ================ Write to the nc file ==========================================
    copy_VarCoords(u(0, :, :), rc_u)
    copy_VarCoords(v(0, :, :), rc_v)
    copy_VarCoords(psl_abnormal(0, :, :), rc_p)
    copy_VarCoords(u(0, :, :),   prob_u_lstc)
    copy_VarCoords(v(0, :, :),   prob_v_lstc)
    copy_VarCoords(psl_abnormal(0, :, :), prob_p_lstc)

    delete(rc_u@nptxy)
    delete(rc_u@rstd)
    delete(rc_u@yintercept)
    delete(rc_u@tval)
    delete(rc_v@nptxy)
    delete(rc_v@rstd)
    delete(rc_v@yintercept)
    delete(rc_v@tval)
    delete(rc_p@nptxy)
    delete(rc_p@rstd)
    delete(rc_p@yintercept)
    delete(rc_p@tval)

;    printMinMax(lstc_detrend, 0)
;    printMinMax(rc_u, 0)
    system("rm -rf /home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_1000to700_avg_moisture_transportation_to_LSTC_ttest.nc")
    fout = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_1000to700_avg_moisture_transportation_to_LSTC_ttest.nc", "c")
    filedimdef(fout,"time",-1,True)
;
    fout->rc_u   = rc_u
    fout->rc_v   = rc_v
    fout->rc_div = rc_p
    fout->p_u  = prob_u_lstc
    fout->p_v  = prob_v_lstc
    fout->p_p  = prob_p_lstc
;
    ;; Set file attribute
    setfileoption(fout,"DefineMode",True)
    fAtt = True
    fAtt@description = "Create on 18/6/2024. This file include regression from qu/qv and divergence to the LSTC index. Created by cal_Anomaly_onset_March_regression_qV_LSTC_index_remove_OLR_240618.ncl"
    fileattdef(fout,fAtt)

end
