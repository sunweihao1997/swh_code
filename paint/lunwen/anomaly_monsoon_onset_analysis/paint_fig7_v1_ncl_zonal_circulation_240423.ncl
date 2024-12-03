;2024-4-23
;This function is to plot the zonal circulation of the regressioned circulation to the OLR index
;use ncl other than py is due to the level axis is always wrong
begin
    f0 = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/regression/ERA5_regression_OLR-remove-SLP_zonal_tropical_circulation.nc", "r")

    u  = f0->rc_u
    w  = f0->rc_w

    level = f0->level
    level@units = "hpa"
    ;printVarSummary(level)

    wAve = avg(w)
    uAve = avg(u)

    scale = fabs(uAve/wAve)

    wscale = w * scale

    copy_VarCoords(w, wscale)

    u&level = level
    wscale&level=level

    wks    = gsn_open_wks("pdf", "/home/sun/paint/monthly_variable/regression/zonal_stream")
    res                   = True                   ; plot mods desired

    ;res@tiMainString      = "Pressure/Height Streamline" ; title
    res@gsnCenterString   = "March zonal circulation regression to OLR index"
   
    res@cnFillOn          = True                ; turn on color fill
    res@cnFillPalette     = "BlAqGrYeOrReVi200" ; choose color map
    res@cnLineLabelsOn    = False               ; turn off line labels

    res@cnLevelSelectionMode = "ExplicitLevels"
    res@cnLevels             = fspan(-3, 3, 11)

    ; 设置X轴和Y轴的范围
    res@trXMinF = 30     ; X轴的最小值
    res@trXMaxF = 150   ; X轴的最大值
;    res@trYMinF = 200   ; Y轴的最小值
;    res@trYMaxF = 1000    ; Y轴的最大值


    res@lbLabelStride     = 2                   ; label every other box
   
    res@stMinArrowSpacingF = 0.008             ; arrow spacing.
    res@stArrowLengthF     = 0.008             ; arrow length

    ; Personal setting

   ;printVarSummary(u)
   ;---draw plot from pole to pole at 170E
    plot  = gsn_csm_pres_hgt_streamline(wks,wscale({1000:100}, :),u({1000:100}, :),\
                                   wscale({1000:100}, :),res )  

end