begin
path  = "/home/sun/data/merra2_climate_vars_multi/daily/"
f0    = addfile(path+"MERRA2_climate_daily_slp_u925_v925.nc","r")
;-------------------------------------------------------
u0    = f0->u925(:,:,:)
v0    = f0->v925(:,:,:)
u0@_FillValue = 9999.99
v0@_FillValue = 9999.99
if(any(isnan_ieee(u0)))then
replace_ieeenan(u0,u0@_FillValue,0)
end if
delete(u0&lat@_FillValue)
if(any(isnan_ieee(v0)))then
replace_ieeenan(v0,v0@_FillValue,0)
end if
delete(v0&lat@_FillValue)

u1           = doubletofloat(u0)
v1           = doubletofloat(v0)
copy_VarCoords(u0, u1)
copy_VarCoords(v0, v1)
u1&lat@units = "degrees_north"
v1&lat@units = "degrees_north"
printVarSummary(u1) 

u2       = dim_avg_n_Wrap(u1(:,:,{80:100}),2)
v2       = dim_avg_n_Wrap(v1(:,:,{80:100}),2)
printVarSummary(u2)
printMinMax(u2,False)

u_draw  = u2(:,{-10:30})
v_draw  = v2(:,{-10:30})
;print(u_draw(:,{28}))
;exit
;*************************
wks  = gsn_open_wks("pdf","/home/sun/paint/phd/phd_c2_liuxl_merra_uv")   ; send graphics to PNG file
gsn_define_colormap(wks,"amwg256")
  res                     = True          ; plot mods desired
  res@gsnDraw             = False
  res@gsnFrame            = False
  res@gsnMaximize         = True          ; maximize plot size
  res@vpWidthF            = 0.5           ; change aspect ratio of plot
  res@vpHeightF           = 0.3
  res@cnFillOn            = True          ; turn on color
  res@cnLevelSelectionMode= "ExplicitLevels"
  res@cnLevels            = fspan(-15., 15., 11)
  res@cnFillColors        = (/45,55,65,75,85,95,175,190,198,200,205,210/)
  res@cnLinesOn           = False
  res@tmXTOn              = False
  res@tmYROn              = False
  
  res@trXMinF             =  0 ;90+0
  res@trXMaxF             = 360  ;122+90
  res@tmXBMode            = "Explicit"
  res@tmXBValues          =(/0,90,180,270,360/)
  res@tmXBLabels          = (/"0","90","180","270","360"/)
  res@tiXAxisString       = "Julian day"
  res@tiYAxisString       = "Latitude"
  res@tmXBLabelDeltaF     = -0.5
  plotA = gsn_csm_contour(wks,u_draw(lat|:,time|:),res)       ; contour the variable
 
  vecres                  = True            ; vector only resources
  vecres@gsnDraw          = False           ; don't draw
  vecres@gsnFrame         = False           ; don't advance frame
  vecres@vcGlyphStyle     = "CurlyVector"   ; curly vectors
  vecres@vcRefMagnitudeF  = 5              ; define vector ref mag
  vecres@vcRefLengthF     = 0.045           ; define length of vec ref
  vecres@gsnRightString   = " "             ; turn off right string
  vecres@gsnLeftString    = " "             ; turn off left string
  vecres@tiXAxisString    = " "             ; turn off axis label 
  vecres@vcRefAnnoOrthogonalPosF = -.505    ; move ref vector into plot
  vecres@vcMinDistanceF   = 0.041
  vecres@vcLineArrowColor       = "grey30"           ; change vector color
  vecres@vcLineArrowThicknessF  = 5.0               ; change vector thickness
  vecres@vcRefAnnoString2On     = False
 
  plotB   = gsn_csm_vector(wks,u_draw(lat|:,time|:),v_draw(lat|:,time|:),vecres)
  overlay(plotA,plotB)                      ; result will be plotA 
 
  draw(plotA)
  frame(wks)

  end