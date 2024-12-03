begin
path = "/home/sun/data/merra2_climate_vars_multi/daily/"
f    = addfile(path+"MERRA2_climate_daily_slp_u925_v925.nc.nc","r")
slp0 = f->slp
slp  = doubletofloat(slp0)
copy_VarCoords(slp0, slp)
printVarSummary(slp)
;print(slp&time)

;April
slpapr  = dim_avg_n_Wrap(slp(90:119,:,:),0)
printVarSummary(slpapr)
printMinMax(slpapr,False)
slpapr  = slpapr /100.0   ; Pa to hPa
slpapr&lat@units = "degrees_north"
slpapr&lon@units = "degrees_east"
;print(slpapr&lat)
;print(slpapr&lon)
;------------------------------------------------------------------------------
wks = gsn_open_wks("pdf", "/home/sun/paint/phd/phd_c4_liuxl_fig_climate_slp_April")
;gsn_define_colormap(wks,"GMT_polar")
gsn_define_colormap(wks,"MPL_RdBu")
res                       = True
res@cnFillOn              = True     ; turn on color fill
res@gsnDraw               = False
res@gsnFrame              = False
res@mpMinLatF             =   0. 
res@mpMaxLatF             =  30.
res@mpMinLonF             =  40.
res@mpMaxLonF             =  120.

res@cnLineLabelsOn        = False
res@cnLinesOn             = False 
;res@lbOrientation         = "Vertical"  
;res@cnInfoLabelOn         = False
res@tmXTOn                = False
res@tmYROn                = False

res@vpKeepAspect          = False
res@vpWidthF              = 0.6    
res@vpHeightF             = 0.65

res@mpGeophysicalLineThicknessF  = 1
res@cnLevelSelectionMode  = "ExplicitLevels" 
res@cnLevels              = ispan(1004, 1013, 1) ;
;res@cnLevels              = (/1004.5, 1005, 1005.5, 1006, 1007, 1008, 1010, 1015, 1020/)
res@cnFillPalette = "MPL_RdBu"
;res@cnFillColors          = (/34,47,74,80,85,95,103,106,109,112,120/) 
;res@cnLevelSpacingF      = 5
res@gsnLeftString             = "SLP"
res@gsnRightString            = "April"
res@gsnLeftStringFontHeightF  = 0.012
res@gsnRightStringFontHeightF = 0.012
res@lbLabelFontHeightF        = 0.012
res@pmLabelBarOrthogonalPosF  = 0.1
res@tmXBLabelFontHeightF      = 0.012
res@tmYLLabelFontHeightF      = 0.012
plot1 = gsn_csm_contour_map(wks,slpapr, res)  ; create plot

draw(plot1)
frame(wks)
end