begin
path  = "/home/sun/data/model_data/climate/b1850_exp/"
f0    = addfile(path+"b1850_maritime_atmosphere_new020.nc","r")
f1    = addfile(path+"b1850_control_climate_atmosphere.nc","r")
;-------------------------------------------------------
pr0    = f0->PRECT(:,:,:)
pr0    = pr0 * 86400000
prc    = f1->PRECT(:,:,:)
prc    = prc * 86400000
pr0@_FillValue = 9999.99
prc@_FillValue = 9999.99

replace_ieeenan(pr0,pr0@_FillValue,0)
replace_ieeenan(prc,prc@_FillValue,0)

delete(pr0&lat@_FillValue)
delete(prc&lat@_FillValue)

pr0    = prc - pr0


pr1       = doubletofloat(pr0)
copy_VarCoords(pr0, pr1)
pr1&lat@units = "degrees_north"
printVarSummary(pr1) 

pr2       = dim_avg_n_Wrap(pr1(:,:,{95:120}),2)
pr_draw   = pr2
pr_draw!0 = "time"
pr_draw!1 = "lat"
printMinMax(pr_draw, False)
pr_draw   = where(pr_draw.lt.-100, pr_draw@_FillValue, pr_draw)
pr_draw   = where(pr_draw.gt.1000, pr_draw@_FillValue, pr_draw)

latavg    = dim_avg_n_Wrap(pr_draw,0)
printVarSummary(latavg)
do ilat   = 0,179
pr_draw(:,ilat)  = where(ismissing(pr_draw(:,ilat)), latavg(ilat) , pr_draw(:,ilat))
;pr_draw(:,ilat)  = where(pr_draw(:,ilat) .eq. , latavg(ilat) , pr_draw(:,ilat))
end do
;print(pr_draw(212, :))
;exit
printVarSummary(pr_draw) ; [time | 365] x [lat | 180]
printMinMax(pr_draw, False)
pr_draw  = runave_n_Wrap(pr_draw, 5, 0, 0)

pr_draw1 = pr_draw(:,{-40:40})
;*************************
wks  = gsn_open_wks("pdf","/home/sun/paint/phd/phd_c2_liuxl_conwomap_maritime_diff")   ; send graphics to PNG file
;wks  = gsn_open_wks("png","conwomap_maritime_diff") 
;gsn_define_colormap(wks,"amwg256")
gsn_define_colormap(wks,"MPL_coolwarm")
  res                     = True          ; plot mods desired
  res@gsnDraw             = False
  res@gsnFrame            = False
  res@gsnMaximize         = True          ; maximize plot size
  res@vpWidthF            = 0.5           ; change aspect ratio of plot
  res@vpHeightF           = 0.3
  res@cnFillOn            = True          ; turn on color
  res@cnLevelSelectionMode= "ExplicitLevels"
  res@cnLevels            = fspan(-5, 5, 11)
  ;res@cnFillColors        = (/190,175,155,145,135,129,80,74,65,55,45,33/)
  
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
  ;res@pmLabelBarOrthogonalPosF = 0.15
 
  plot = gsn_csm_contour(wks,pr_draw1(lat|:,time|:),res)       ; contour the variable
  draw(plot)
  frame(wks)

  end