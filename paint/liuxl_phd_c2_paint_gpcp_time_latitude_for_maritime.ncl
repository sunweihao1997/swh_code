begin
path  = "/home/sun/data/model_data/climate/b1850_exp/"
f0    = addfile(path+"b1850_maritime_atmosphere_new020.nc","r")
;-------------------------------------------------------
pr0    = f0->PRECT(:,:,:)
pr0    = pr0 * 86400000
pr0@_FillValue = 9999.99
if(any(isnan_ieee(pr0)))then
replace_ieeenan(pr0,pr0@_FillValue,0)
end if
delete(pr0&lat@_FillValue)

pr1       = doubletofloat(pr0)
copy_VarCoords(pr0, pr1)
pr1&lat@units = "degrees_north"
printVarSummary(pr1) 

pr2       = dim_avg_n_Wrap(pr1(:,:,{105:130}),2)
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
wks  = gsn_open_wks("png","conwomap_maritime2")   ; send graphics to PNG file
gsn_define_colormap(wks,"amwg256")
  res                     = True          ; plot mods desired
  res@gsnDraw             = False
  res@gsnFrame            = False
  res@gsnMaximize         = True          ; maximize plot size
  res@vpWidthF            = 0.5           ; change aspect ratio of plot
  res@vpHeightF           = 0.3
  res@cnFillOn            = True          ; turn on color
  res@cnLevelSelectionMode= "ExplicitLevels"
  res@cnLevels            = fspan(3., 18., 6)
  res@cnFillColors        = (/0,175,190,198,200,205,210/)
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
  plot = gsn_csm_contour(wks,pr_draw1(lat|:,time|:),res)       ; contour the variable
  draw(plot)
  frame(wks)

  end