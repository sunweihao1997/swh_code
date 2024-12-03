begin
path = "/home/sun/data/model_data/climate/b1850_exp/"
f    = addfile(path+"b1850_control_climate_atmosphere.nc","r")
f1   = addfile(path+"b1850_maritime_climate_atmosphere.nc","r")
;-------------------------- control --------------------------------
u_tmp = f->U
w_tmp = f->OMEGA
value = 1.e20
if (any(isnan_ieee(u_tmp))) then
replace_ieeenan(u_tmp, value, 0)
end if
u_tmp@_FillValue = value
if (any(isnan_ieee(w_tmp))) then
replace_ieeenan(w_tmp, value, 0)
end if
u_tmp@_FillValue = value
w_tmp@_FillValue = value
;printVarSummary(u_tmp) ;[time | 365] x [lev | 29] x [lat | 192] x [lon | 288]
print(u_tmp&lev)

;-------- time & Lon 105:120 avg-------------
uapr1   = dim_avg_n_Wrap(u_tmp(:,{1000:200},{-20:20},{105:120}),3)   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
uapr2   = dim_avg_n_Wrap(uapr1(95:127,:,:),0)
uapr3   = dim_avg_n_Wrap(uapr1(105:120,:,:),0)
printVarSummary(uapr1)
delete([/u_tmp,uapr1/])
uapr2&lev@long_name = "Presure"
uapr2&lev@units     = "hPa"
uapr2&lat@units     = "degrees_north"
uapr3&lev@long_name = "Presure"
uapr3&lev@units     = "hPa"
uapr3&lat@units     = "degrees_north"

wapr1   = dim_avg_n_Wrap(w_tmp(:,{1000:200},{-20:20},{105:120}),3)   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
wapr2   = dim_avg_n_Wrap(wapr1(95:127,:,:),0)
wapr3   = dim_avg_n_Wrap(wapr1(105:120,:,:),0)
printVarSummary(wapr1)
delete([/w_tmp,wapr1/])
wapr2&lev@long_name = "Presure"
wapr2&lev@units     = "hPa"
wapr2&lat@units     = "degrees_north"
wapr3&lev@long_name = "Presure"
wapr3&lev@units     = "hPa"
wapr3&lat@units     = "degrees_north"
;---------------------- sensitivity ----------------------
u_tmp = f1->U
w_tmp = f1->OMEGA
value = 1.e20
if (any(isnan_ieee(u_tmp))) then
replace_ieeenan(u_tmp, value, 0)
end if
u_tmp@_FillValue = value
if (any(isnan_ieee(w_tmp))) then
replace_ieeenan(w_tmp, value, 0)
end if
u_tmp@_FillValue = value
w_tmp@_FillValue = value
printVarSummary(u_tmp) ;[time | 365] x [lev | 29] x [lat | 192] x [lon | 288]
;-------- April & Lon 40:120 avg-------------
uapr1       = dim_avg_n_Wrap(u_tmp(:,{1000:200},{-20:20},{105:120}),3)    ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
uapr2_sen   = dim_avg_n_Wrap(uapr1(95:127,:,:),0)
uapr3_sen   = dim_avg_n_Wrap(uapr1(105:120,:,:),0)
delete([/u_tmp,uapr1/])
uapr2_sen&lev@long_name = "Presure"
uapr2_sen&lev@units     = "hPa"
uapr2_sen&lat@units     = "degrees_north"
uapr3_sen&lev@long_name = "Presure"
uapr3_sen&lev@units     = "hPa"
uapr3_sen&lat@units     = "degrees_north"

wapr1       = dim_avg_n_Wrap(w_tmp(:,{1000:200},{-20:20},{105:120}),3)   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
wapr2_sen   = dim_avg_n_Wrap(wapr1(95:127,:,:),0)
wapr3_sen   = dim_avg_n_Wrap(wapr1(105:120,:,:),0)
delete([/w_tmp,wapr1/])
wapr2_sen&lev@long_name = "Presure"
wapr2_sen&lev@units     = "hPa"
wapr2_sen&lat@units     = "degrees_north"
wapr3_sen&lev@long_name = "Presure"
wapr3_sen&lev@units     = "hPa"
wapr3_sen&lat@units     = "degrees_north"

;print(uapr2({500},:))
;print(uapr2_sen({500},:))
;print(wapr2({500},:))
;print(wapr2_sen({500},:))

;----------- to draw  ---------------
; (uapr2, wapr2)  (uapr3, wapr3)   (uapr2_sen,wapr2_sen) (uapr3_sen,wapr3_sen)

wapr2_scale     = wapr2 * 1000. ;scale
wapr3_scale     = wapr3 * 1000.
wapr2_sen_scale = wapr2_sen * 1000.
wapr3_sen_scale = wapr3_sen * 1000.

wapr2_scale!0             = "lev"
wapr2_scale!1             = "lat"
wapr2_scale&lev           = tofloat(uapr2&lev)
wapr2_scale&lev@units     = "hPa"
wapr2_scale&lat           = uapr2&lat
copy_VarCoords(wapr2_scale, wapr3_scale)
copy_VarCoords(wapr2_scale, wapr2_sen_scale)
copy_VarCoords(wapr2_scale, wapr3_sen_scale)

;----- diff ----------
uapr2_diff     = uapr2 - uapr2_sen
uapr3_diff     = uapr3 - uapr3_sen
wapr2_scale_diff     = wapr2_scale - wapr2_sen_scale
wapr3_scale_diff     = wapr3_scale - wapr3_sen_scale
copy_VarCoords(wapr2_scale, wapr2_scale_diff)
copy_VarCoords(wapr2_scale, uapr2_diff)
copy_VarCoords(wapr2_scale, wapr3_scale_diff)
copy_VarCoords(wapr2_scale, uapr3_diff)

;print(wapr2_scale_diff({500},:))
;exit
;print(wapr2_sen({500},:))
;----------------------------------------------------------
  wks   = gsn_open_wks ("pdf","/home/sun/paint/phd/phd_c4_liuxl_control_self_maritime_streamfunction")
  plot  = new(6,graphic) 
  res                      = True                   ; plot mods desired
  res@gsnDraw              = False    
  res@gsnFrame             = False
  ;res@tiMainString        = "Pressure/Height Streamline" ; title
  res@gsnCenterString      = "  "
  res@cnFillOn             = True                ; turn on color fill
  res@cnFillPalette        = "MPL_RdBu" ; choose color map
  res@cnLineLabelsOn       = False               ; turn off line labels
  res@cnLinesOn            = False
  res@cnLevelSelectionMode = "ExplicitLevels"
  res@stLineThicknessF     = 2.
  ;res@cnLevels             = fspan(-25, 25, 11)
  res@cnLevels             = (/-90,-80,-70,-60,-40,-20,-10,0,10,20,40,60,70,80,90/)
  res@cnFillColors         = (/13,17,24,26,32,35,42,50,73,78,82,89,97,102,113,116/)
;  res@trXMinF              = -20     ; X轴的最小值
;  res@trXMaxF              = 20   ; X轴的最大值
; res@trYMinF              = 200   ; Y轴的最小值
; res@trYMaxF              = 1000    ; Y轴的最大值
  res@lbLabelStride        = 2                  ; label every other box
  res@stMinArrowSpacingF   = 0.008             ; arrow spacing.
  res@stArrowLengthF       = 0.008             ; arrow length
  res@tmXTOn               = False
  res@tmYRMode             = "Automatic"
  res@tmYROn               = False
  res@gsnRightString       = "control"
  res@gsnLeftString        = "90-110"
  plot(0)  = gsn_csm_pres_hgt_streamline(wks,wapr2_scale(:,:),uapr2(:,:),wapr2_scale(:,:),res ) 
  res@gsnLeftString        = "110-130" 
  plot(1)  = gsn_csm_pres_hgt_streamline(wks,wapr3_scale(:,:),uapr3(:,:),wapr3_scale(:,:),res ) 

  res@gsnRightString       = "maritime"
  res@gsnLeftString        = "90-110"
  plot(2)  = gsn_csm_pres_hgt_streamline(wks,wapr2_sen_scale(:,:),uapr2_sen(:,:),wapr2_sen_scale(:,:),res ) 
  res@gsnLeftString        = "110-130" 
  plot(3)  = gsn_csm_pres_hgt_streamline(wks,wapr3_sen_scale(:,:),uapr3_sen(:,:),wapr3_sen_scale(:,:),res ) 

  res@gsnRightString       = "control_maritime"
  res@gsnLeftString        = "90-110" 
  plot(4)  = gsn_csm_pres_hgt_streamline(wks,wapr2_scale_diff(:,:),uapr2_diff(:,:),wapr2_scale_diff(:,:),res ) 
  res@gsnLeftString        = "110-130" 
  plot(5)  = gsn_csm_pres_hgt_streamline(wks,wapr3_scale_diff(:,:),uapr3_diff(:,:),wapr3_scale_diff(:,:),res ) 
  gsn_panel(wks,plot,(/3,2/),False)    
  end