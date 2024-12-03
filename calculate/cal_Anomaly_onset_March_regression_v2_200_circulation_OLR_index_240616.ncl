;2023-5-5
;This script calculate the regression of wind to the OLR
begin
    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================
    ; Wind data
    fvar = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/monthly/ERA5_1980_2021_monthly_200hpa_UVZ_500hpa_w.nc", "r")


    ; Land-Sea thermal contrast
    f1 = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_month/ERA5_LSTC_1980_2021_t2m_ts_sp_psl_mar.nc", "r")

    ; Read data
    u  = fvar->Mar_u
    v  = fvar->Mar_v
    ;psl= fvar->Mar_z
    w  = fvar->Mar_w
    t  = fvar->Mar_t

    lat= fvar->lat
    lon= fvar->lon

    lstc = f1->OLR_mari_Afri


;    print(lstc)

    ; Calculates and removes the mean of the given dimension(s) at all other dimensions.
    lstc_abnormal = dim_rmvmean_n_Wrap(lstc, 0)
    u_abnormal    = dim_rmvmean_n_Wrap(u, 0)
    v_abnormal    = dim_rmvmean_n_Wrap(v, 0)
    w_abnormal    = dim_rmvmean_n_Wrap(w, 0)
    t_abnormal    = dim_rmvmean_n_Wrap(t, 0)
    ;psl_abnormal  = dim_rmvmean_n_Wrap(psl/9.8,0)
    ;printVarSummary(psl_abnormal)
    ;exit

    ;print(u_abnormal)
    ; Detrend
    lstc_detrend  = lstc_abnormal
    lstc_detrend  = (/dim_standardize_n_Wrap(dtrend_n(lstc_abnormal, True, 0), 1, 0)/)

    u_detrend     = u_abnormal
    u_detrend     = (/dtrend_n(u_abnormal, True, 0)/)
    v_detrend     = v_abnormal
    v_detrend     = (/dtrend_n(v_abnormal, True, 0)/)
    w_detrend     = w_abnormal
    w_detrend     = (/dtrend_n(w_abnormal, True, 0)/)
    t_detrend     = t_abnormal
    t_detrend     = (/dtrend_n(t_abnormal, True, 0)/)
    ;p_detrend     = psl_abnormal
    ;p_detrend     = (/dtrend_n(psl_abnormal, True, 0)/)




    ; Change the dimension sequence
    ;printVarSummary(u_abnormal)
    u_detrend_    = u_detrend
    v_detrend_    = v_detrend
    w_detrend_    = w_detrend
    t_detrend_    = t_detrend
    ;p_detrend_    = p_detrend
    ;printVarSummary(lstc_detrend)
    ;printVarSummary(u_detrend_)

    ; Delete NAN value
    value = 1.e20
    replace_ieeenan(u_detrend_, value, 0)
    replace_ieeenan(v_detrend_, value, 0)
    replace_ieeenan(w_detrend_, value, 0)
    replace_ieeenan(t_detrend_, value, 0)
    ;replace_ieeenan(p_detrend_, value, 0)
    replace_ieeenan(lstc_detrend, value, 0)
    u_detrend_@_FillValue = value
    v_detrend_@_FillValue = value
    t_detrend_@_FillValue = value
    w_detrend_@_FillValue = value
    lstc_detrend@_FillValue = value

    ; ------ calculate temp advection -----
    temp_grad     = grad_latlon_cfd(t_detrend, lat, lon, True, False)
    temp_grad_lon = temp_grad[1] ;zonal
    temp_grad_lat = temp_grad[0] ;meridional
  
    temp_adv      = -(u_detrend*temp_grad_lon + v_detrend*temp_grad_lat)

    temp_adv_detrend_    = temp_adv

    copy_VarCoords(t_detrend_, temp_adv_detrend_)

    ; ------ calculate uv divergence ----
    div           = uv2dv_cfd(u_detrend_, v_detrend_, lat, lon, 2)

    div_detrend_        = div
    copy_VarCoords(t_detrend_, div_detrend_)


    ;=================================================
    ; ================ Data processing1: abnormal, detrend, calculate monthly mean =======================

    ; ================ Data processing2: calculate regression =======================
    ;printVarSummary(u_detrend)
    ;printVarSummary(lstc_detrend_april)
    rc_u = regCoef_n(lstc_detrend, u_detrend_, 0, 0)  
    rc_v = regCoef_n(lstc_detrend, v_detrend_, 0, 0)  
    rc_t = regCoef_n(lstc_detrend, t_detrend_, 0, 0)
    rc_w = regCoef_n(lstc_detrend, w_detrend_, 0, 0)
    rc_a = regCoef_n(lstc_detrend, temp_adv_detrend_, 0, 0) ; advection
    rc_d = regCoef_n(lstc_detrend, div_detrend_, 0, 0) ; divergence

    ;Calculate the correlation
    r_u_lstc = escorc(lstc_detrend,u_detrend_(lat|:,lon|:,time|:))
    r_v_lstc = escorc(lstc_detrend,v_detrend_(lat|:,lon|:,time|:))
    ;printVarSummary(r_u_lstc)
    r_t_lstc = escorc(lstc_detrend,t_detrend_(lat|:,lon|:,time|:))
    r_w_lstc = escorc(lstc_detrend,w_detrend_(lat|:,lon|:,time|:))
    r_a_lstc = escorc(lstc_detrend,temp_adv_detrend_(lat|:,lon|:,time|:))
    r_d_lstc = escorc(lstc_detrend,div_detrend_(lat|:,lon|:,time|:))
    copy_VarCoords(u_detrend_(0,:,:),r_u_lstc)
    copy_VarCoords(v_detrend_(0,:,:),r_v_lstc)
    copy_VarCoords(t_detrend_(0,:,:),r_t_lstc)
    copy_VarCoords(w_detrend_(0,:,:),r_w_lstc)
    copy_VarCoords(w_detrend_(0,:,:),r_a_lstc)
    copy_VarCoords(w_detrend_(0,:,:),r_d_lstc)

;    Nr       = dimsizes(lstc_detrend)
;    t_u      = r_u_lstc*sqrt((Nr-2)/(1-r_u_lstc^2))
;    t_v      = r_v_lstc*sqrt((Nr-2)/(1-r_v_lstc^2))
;    t_t      = r_t_lstc*sqrt((Nr-2)/(1-r_t_lstc^2))
;    t_w      = r_w_lstc*sqrt((Nr-2)/(1-r_w_lstc^2))
;    t_a      = r_a_lstc*sqrt((Nr-2)/(1-r_w_lstc^2))
;    t_d      = r_d_lstc*sqrt((Nr-2)/(1-r_w_lstc^2))
    ;test significant
    Nr          = dimsizes(lstc_detrend)
    prob_u_lstc = rtest(r_u_lstc , Nr, 0)
    prob_v_lstc = rtest(r_v_lstc , Nr, 0)
    prob_w_lstc = rtest(r_w_lstc , Nr, 0)
    prob_t_lstc = rtest(r_t_lstc , Nr, 0)
    prob_a_lstc = rtest(r_a_lstc , Nr, 0)
    prob_d_lstc = rtest(r_d_lstc , Nr, 0)

    ; ================ Write to the nc file ==========================================
    copy_VarCoords(u(0, :, :), rc_u)
    copy_VarCoords(v(0, :, :), rc_v)
    copy_VarCoords(w(0, :, :), rc_w)
    copy_VarCoords(t(0, :, :), rc_t)
    copy_VarCoords(t(0, :, :), rc_a)
    copy_VarCoords(t(0, :, :), rc_d)

    copy_VarCoords(u(0, :, :),   prob_u_lstc)
    copy_VarCoords(v(0, :, :),   prob_v_lstc)
    copy_VarCoords(w(0, :, :),   prob_w_lstc)
    copy_VarCoords(t(0, :, :),   prob_t_lstc)
    copy_VarCoords(t(0, :, :),   prob_a_lstc)
    copy_VarCoords(t(0, :, :),   prob_d_lstc)

    delete(rc_u@nptxy)
    delete(rc_u@rstd)
    delete(rc_u@yintercept)
    delete(rc_u@tval)
    delete(rc_v@nptxy)
    delete(rc_v@rstd)
    delete(rc_v@yintercept)
    delete(rc_v@tval)
    delete(rc_t@nptxy)
    delete(rc_t@rstd)
    delete(rc_t@yintercept)
    delete(rc_t@tval)
    delete(rc_w@nptxy)
    delete(rc_w@rstd)
    delete(rc_w@yintercept)
    delete(rc_w@tval)

;    printMinMax(lstc_detrend, 0)
;    printMinMax(rc_u, 0)
    system("rm -rf /home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_200_uvz_500_w_to_OLR_ttest.nc")
    fout = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_200_uvz_500_w_to_OLR_ttest.nc", "c")
    filedimdef(fout,"time",-1,True)
;
    fout->rc_u = rc_u
    fout->rc_v = rc_v
    fout->rc_w = rc_w
    fout->rc_t = rc_t
    fout->rc_a = rc_a
    fout->rc_d = rc_d
 
   
    fout->prob_u_olr = prob_u_lstc
    fout->prob_v_olr = prob_v_lstc
    fout->prob_w_olr = prob_w_lstc
    fout->prob_t_olr = prob_t_lstc
    fout->prob_a_olr = prob_a_lstc
    fout->prob_d_olr = prob_d_lstc

;
    ;; Set file attribute
    setfileoption(fout,"DefineMode",True)
    fAtt = True
    fAtt@description = "Create on 17/6/2024. This file include regression from u/v/t at 200 hPa and w at 500 hPa to the OLR index. Created by cal_Anomaly_onset_March_regression_200_circulation_OLR_index_240616.ncl"
    fileattdef(fout,fAtt)

end
