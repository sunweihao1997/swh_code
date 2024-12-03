begin
path = "/home/sun/data/merra2_climate_vars_multi/daily/"
f    = addfile(path+"MERRA2_climate_daily_slp_u925_v925.nc.nc","r")
slp0 = f->slp
slp  = doubletofloat(slp0)
copy_VarCoords(slp0, slp)
printVarSummary(slp)
slp  = slp/100.0   ; Pa to hPa
slp&lat@units = "degrees_north"
slp&lon@units = "degrees_east"
delete(slp0)
;------------cross equ flow----------------------
v925_tmp = f->v925
v925     = doubletofloat(v925_tmp)
copy_VarCoords(v925_tmp, v925)
printVarSummary(v925)
delete(v925_tmp)
;------------------------------------------------
;Pentad
tsIndia_tmp  = dim_avg_n_Wrap(slp(:,{5:20},{70:90}),(/1,2/)) 
tsTIO_tmp    = dim_avg_n_Wrap(slp(:,{-10:10},{55:95}),(/1,2/)) 
CEF_tmp      = dim_avg_n_Wrap(v925(:,{-5:5},{80:90}),(/1,2/))
delete(slp)
delete(v925)

tsIndia      = new(73,"float")
tsTIO        = new(73,"float")
CEF          = new(73,"float")
do ip        = 1-1,73-1,1 
tsIndia(ip)  = dim_avg(tsIndia_tmp(ip*5+0: ip*5+4))
tsTIO(ip)    = dim_avg(tsTIO_tmp(ip*5+0  : ip*5+4))
CEF(ip)      = dim_avg(CEF_tmp(ip*5+0  : ip*5+4))
end do 

tsIndia!0      = "pentad"
tsIndia&pentad = ispan(1,73,1)
tsTIO!0        = "pentad"
tsTIO&pentad   = ispan(1,73,1)
CEF!0          = "pentad"
CEF&pentad     = ispan(1,73,1)
;printMinMax(CEF(14:24), False)

data        = new( (/2,73/),float)
data(0,:)   = (/tsIndia/)
data(1,:)   = (/tsTIO/)
data!1      = "pentad"
data&pentad = ispan(1,73,1)

;------------------------------------------
wks   = gsn_open_wks ("pdf","/home/sun/paint/phd/phd_c4_liuxl_Ts_Indian_TIO_CEF_P15_25")               
              
resL                        = True 
resL@gsnDraw                = False                   ; don't draw
resL@gsnFrame               = False	
resL@xyLineThicknesses      = (/5.5,5.0/)
resL@trYMaxF                = 1018.                
resL@trYMinF                = 1003. 
resL@tiYAxisString          = "Sea Level Pressure (hPa)"
resL@tiXAxisString          = "Pentad"
resL@tiMainString           = "  "
resL@xyExplicitLegendLabels = (/"India","TIO"/)        
resL@xyLineColors           = (/"red","red"/) 
resL@xyDashPatterns         =(/0, 2/)
resL@tmXTOn                 = False
resL@trXMinF                = 14
resL@trXMaxF                = 26
resL@tmXBMode               ="Explicit"
resL@tmXBValues             =(/14,15,16,17,18,19,20,21,22,23,24,25,26/)
resL@tmXBLabels             ="" + resL@tmXBValues

resR                        = True
resR@trYMaxF                = 7.                    ; axis max
resR@trYMinF                = -7.                    ; axis min
resR@xyLineColors           = "blue"                    ; line color
resR@tiYAxisString          = "CEF(m/s)"
resR@xyLineThicknesses      = 5.5
;plot=gsn_csm_xy(wks,data&pentad(14:24),data(:,14:24),res)
;plot = gsn_csm_xy2(wks,time,t,p,resL,resR)
plot = gsn_csm_xy2(wks,data&pentad(14:24),data(:,14:24),CEF(14:24),resL,resR )


; legend resources
;***********************************************
labels                   = (/"India","TIO","CEF"/)
lgres                    = True
lgres@lgLineColors       = (/"red","red","blue"/)    ; colors for legend lines
lgres@vpWidthF           = 0.2                   ; width of legend (NDC)
lgres@vpHeightF          = 0.1                   ; height of legend (NDC)
lgres@lgPerimOn          = False                 ; turn off perimeter
lgres@lgDashIndexes      = (/0,2,0/)              ; dash indexes
lgres@lgLineThicknessF   = resR@xyLineThicknesses
lgres@lgLineLabelStrings = labels             ; no labels in lines
gsn_legend_ndc(wks,3,labels,0.55,0.78,lgres)
draw(plot)
frame(wks)
end