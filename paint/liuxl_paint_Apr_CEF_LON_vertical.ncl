begin
path = "/home/sun/data/merra2_climate_vars_multi/monthly/"
f    = addfile(path+"MERRA2_climate_daily_v.nc","r")
;------------ cross equ flow ----------------------
v_tmp = f->v
value = 1.e20
if (any(isnan_ieee(v_tmp))) then
replace_ieeenan(v_tmp, value, 0)
end if
v_tmp@_FillValue = value

v     = doubletofloat(v_tmp)
copy_VarCoords(v_tmp, v)
printVarSummary(v) ;[time | 365] x [lev | 17] x [lat | 361] x [lon | 576]
delete(v_tmp)

;April & Lon 40:120 avg
;vapr  = dim_avg_n_Wrap(v(90:119,:,:,{40:120}),(/0,3/))
vapr1  = v(:,:,{0},{50:120})   ; EQ   ;dim_avg_n_Wrap(v(:,:,:,{40:120}),3)
vapr2  = dim_avg_n_Wrap(vapr1(90:119,:,:),0)
printVarSummary(vapr2)
printMinMax(vapr2,False)

;vapr2&lat@units     = "degrees_north"
vapr2&lev@long_name = "Presure"
vapr2&lev@units     = "hPa"
vapr2&lon@units = "degrees_east"
;----------------------------------------------------------
  wks   = gsn_open_wks ("pdf","/home/sun/paint/phd/phd_c4_liuxl_CEF_LON_vertical" )          ; send graphics to PNG file
  res                      = True                 ; plot mods desired
  res@cnLevelSelectionMode = "ManualLevels"       ; manually select levels
  res@cnLevelSpacingF      = 0.5                  ; contour spacing
  res@cnMinLevelValF       = -5.                 ; min level
  res@cnMaxLevelValF       =  5.                 ; max level
  res@cnLineLabelsOn       = False                 ; turn on line labels
  res@cnFillOn             = True                 ; turn on color fill
  res@cnFillPalette        = "BlWhRe"             ; choose colormap
  res@tiYAxisString        = vapr2&lev@long_name + " (" + vapr2&lev@units + ")"
  res@tmXTOn               = False
  res@tmYRMode             = "Automatic"
  res@tmYROn               = False
  res@tmXBMode             = "Manual"	
  res@tmXBTickStartF       = 50
  res@tmXBTickEndF         = 120
  res@tmXBTickSpacingF     = 10
  ;;res@mpShapeMode          = "FreeAspect"
  ;res@vpXF                 = 0.25
  ;res@vpYF                 = 0.8
  ;res@vpWidthF             = 0.5
  ;res@vpHeightF            = 0.3

  ;res@tmYROn               = False
  ;res@tmYRLabels           = " "
  ;drawNDCGrid(wks)
  plot  = gsn_csm_pres_hgt(wks, vapr2, res )   ; place holderß
  end