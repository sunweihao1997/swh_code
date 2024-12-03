begin
path = "/home/sun/data/model_data/climate/b1850_exp/"
f    = addfile(path+"b1850_control_climate_atmosphere.nc","r")
f1   = addfile(path+"b1850_inch_climate_atmosphere.nc","r")
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
;-------- April & Lon 40:120 avg-------------
uapr1   = dim_avg_n_Wrap(u_tmp(:,:,{10:20},{70:105}),2)   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
uapr2   = dim_avg_n_Wrap(uapr1(90:119,:,:),0)
printVarSummary(uapr1)
delete([/u_tmp,uapr1/])
uapr2&lev@long_name = "Presure"
uapr2&lev@units     = "hPa"
uapr2&lon@units     = "degrees_east"

wapr1   = dim_avg_n_Wrap(w_tmp(:,:,{10:20},{70:105}),2)   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
wapr2   = dim_avg_n_Wrap(wapr1(90:119,:,:),0)
printVarSummary(wapr1)
delete([/w_tmp,wapr1/])
wapr2&lev@long_name = "Presure"
wapr2&lev@units     = "hPa"
wapr2&lon@units     = "degrees_east"
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
;printVarSummary(u_tmp) ;[time | 365] x [lev | 29] x [lat | 192] x [lon | 288]
;-------- April & Lon 40:120 avg-------------
uapr1       = dim_avg_n_Wrap(u_tmp(:,:,{10:20},{70:105}),2)   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
uapr2_sen   = dim_avg_n_Wrap(uapr1(90:119,:,:),0)
delete([/u_tmp,uapr1/])
uapr2_sen&lev@long_name = "Presure"
uapr2_sen&lev@units     = "hPa"
uapr2_sen&lon@units     = "degrees_east"

wapr1       = dim_avg_n_Wrap(w_tmp(:,:,{10:20},{70:105}),2)   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
wapr2_sen   = dim_avg_n_Wrap(wapr1(90:119,:,:),0)
delete([/w_tmp,wapr1/])

wapr2_sen&lev@long_name = "Presure"
wapr2_sen&lev@units     = "hPa"
wapr2_sen&lon@units     = "degrees_east"
;----------- cal diff ---------------
;u      = uapr2
u      = uapr2 - uapr2_sen ;;[lev | 29] x [lon | 97]
w      = wapr2 - wapr2_sen
;printVarSummary(wapr2)
;printVarSummary(wapr2_sen)
;;w      = wapr2 - wapr2_sen
;wAve   = avg(w)
;uAve   = avg(u)
;printVarSummary(uAve) ;[1]

;scale  = fabs(uAve/wAve)
wscale = w * 1000 ;scale

wscale!0 = "lev"
wscale!1 = "lon"
wscale&lev = tofloat(uapr2&lev)
wscale&lev@units     = "hPa"
wscale&lon = uapr2&lon
copy_VarCoords(wscale, u)
;copy_VarCoords(wscale, wscale)
;copy_VarCoords(w, wscale)
;printVarSummary(wscale) 
;printMinMax(wscale, False) ; min=-40.2803   max=9.19533
;printVarSummary(u)    ;[lev | 29] x [lon | 97]
;printMinMax(u, False) ;min=-2.41217   max=3.10875
;
;print(wscale&lev) ;int64
;print(u&lev)

;print(wscale&lev)
;exit
;----------------------------------------------------------
  wks   = gsn_open_wks ("pdf","/home/sun/paint/phd/phd_c4_v2_con_minus_sen_zonal_stream_LON_vertical_maritime" )

  res                      = True                   ; plot mods desired
  ;res@tiMainString        = "Pressure/Height Streamline" ; title
  res@gsnCenterString      = "  "
  res@cnFillOn             = True                ; turn on color fill
  res@cnFillPalette        = "MPL_RdBu" ; choose color map
  res@cnLineLabelsOn       = False               ; turn off line labels
  res@cnLinesOn            =False
  res@cnLevelSelectionMode = "ExplicitLevels"
  ;res@cnLevels             = fspan(-25, 25, 11)
  res@cnLevels             = (/-40, -30, -20,-10, -5, 0, 5,10,20,30, 40/)
  ;res@cnFillColors         = (/13,17,24,26,32,35,42,50,73,78,82,89,97,102,113,116/)
  res@cnFillColors         = (/13,26,32,35,42,50,73,78,82,89,97,102/)
  res@trXMinF              = 70     ; X轴的最小值
  res@trXMaxF              = 105   ; X轴的最大值
; res@trYMinF              = 200   ; Y轴的最小值
; res@trYMaxF              = 1000    ; Y轴的最大值
  res@lbLabelStride        = 2                  ; label every other box
  res@stMinArrowSpacingF   = 0.008             ; arrow spacing.
  res@stArrowLengthF       = 0.008             ; arrow length
  res@tmXTOn               = False
  res@tmYRMode             = "Automatic"
  res@tmYROn               = False
  plot  = gsn_csm_pres_hgt_streamline(wks,wscale({1000:500},:),u({1000:500},:),wscale({1000:500}, :),res )  

  end