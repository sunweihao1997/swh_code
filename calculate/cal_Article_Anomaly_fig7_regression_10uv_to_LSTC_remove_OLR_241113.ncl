;2023-5-5
;This script calculate the regression of wind to the OLR
begin
    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================
    ; Wind data
    fpsl = addfile("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_psl_mar.nc", "r")
    fu10 = addfile("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_u10_mar.nc", "r")
    fv10 = addfile("/home/sun/mydown/ERA5/monthly_single/process/1980_2021_multiple_year_single_v10_mar.nc", "r")

    ; Land-Sea thermal contrast
    f1 = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_mar.nc", "r")

    ; Read data
    u  = fu10->u10
    v  = fv10->v10
    psl= fpsl->msl

    lstc_remove        = f1->LSTC_psl_IOB
    olr                = f1->OLR_mari_Afri

    ;print(lstc_remove)

    
    ; remove the LSTC signal
    regh1=new(1,"float",0)
    tval1=new(1,"float",0)
    nptxy1=new(1,"integer",0)
    b1            = regcoef(olr, lstc_remove, tval1, nptxy1)
    lstc_mean      = dim_avg(lstc_remove)
    a1            = lstc_mean - b1*dim_avg(olr)

    lstc_remove_olr = lstc_remove - (b1 * olr + a1)

    lstc_remove_olr_standardize = dim_standardize(lstc_remove_olr, 1)
    print(lstc_remove_olr_standardize)
    exit

    ; Calculates and removes the mean of the given dimension(s) at all other dimensions.
    lstc_abnormal = dim_rmvmean_n_Wrap(lstc_remove_olr, 0)
    u_abnormal    = dim_rmvmean_n_Wrap(u, 0)
    v_abnormal    = dim_rmvmean_n_Wrap(v, 0)
    psl_abnormal  = dim_rmvmean_n_Wrap(psl,0)

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

    ; Delete NAN value
    value = 1.e20
    replace_ieeenan(u_detrend_, value, 0)
    replace_ieeenan(v_detrend_, value, 0)
    replace_ieeenan(p_detrend_, value, 0)
    replace_ieeenan(lstc_detrend, value, 0)
    u_detrend_@_FillValue = value
    v_detrend_@_FillValue = value
    p_detrend_@_FillValue = value
    lstc_detrend@_FillValue = value



    ;=================================================
    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================

    ; ================ Data processing2: calculate regression =======================
    ;printVarSummary(u_detrend)
    ;printVarSummary(lstc_detrend_april)
    lstc_detrend!0    = "time"
    lstc_detrend&time = u_detrend_&time
    
    rc_u = regCoef_n(lstc_detrend, u_detrend_, 0, 0)  
    rc_v = regCoef_n(lstc_detrend, v_detrend_, 0, 0)  
    rc_p = regCoef_n(lstc_detrend, p_detrend_, 0, 0)

    ;Calculate the correlation
    r_u_lstc = escorc(lstc_detrend,u_detrend_(latitude|:,longitude|:,time|:))
    r_v_lstc = escorc(lstc_detrend,v_detrend_(latitude|:,longitude|:,time|:))
    r_p_lstc = escorc(lstc_detrend,p_detrend_(latitude|:,longitude|:,time|:))
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
;    printVarSummary(u_detrend_)
    ;print(prob_u_lstc(370, :))
;    print(r_u_lstc)

;    exit
    ; ================ Write to the nc file ==========================================
    copy_VarCoords(u(0, :, :), rc_u)
    copy_VarCoords(v(0, :, :), rc_v)
    copy_VarCoords(psl(0, :, :), rc_p)
    copy_VarCoords(u(0, :, :),   prob_u_lstc)
    copy_VarCoords(v(0, :, :),   prob_v_lstc)
    copy_VarCoords(psl(0, :, :), prob_p_lstc)

    ;exit

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
    ;system("rm -rf /home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_10uv_psl_to_LSTC_ttest.nc")
    fout = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_10uv_psl_to_LSTC_remove_OLR_ttest_March.nc", "c")
    filedimdef(fout,"time",-1,True)
;
    fout->rc_u = rc_u
    fout->rc_v = rc_v
    fout->rc_p = rc_p
    fout->p_u  = prob_u_lstc
    fout->p_v  = prob_v_lstc
    fout->p_p  = prob_p_lstc
;
    ;; Set file attribute
    setfileoption(fout,"DefineMode",True)
    fAtt = True
    fAtt@description = "Create on 13/Nov/2024 by /home/sun/swh_code/calculate/cal_Article_Anomaly_fig7_regression_10uv_to_LSTC_remove_OLR_241113.ncl. This file include regression from u/v to the LSTC index, while the index of LSTC has been standardized."
    fileattdef(fout,fAtt)

end
