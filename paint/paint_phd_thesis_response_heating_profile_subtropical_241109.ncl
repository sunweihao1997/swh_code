;2024-11-09
;This script is to response to the sb reviewer
;本代码绘制80E逐侯的加热廓线
begin
path  =  "/home/sun/qomo-data/"
path2 =  "/home/sun/paint/phd/"

f  =  addfile(path+"composite-heating-merra.nc","r")
physics      = f->physics({23:30},:,{20:25},{90:95})
turbulence = f->turbulence({23:30},:,{20:25},{90:95})
moist      = f->moist({23:30},:,{20:25},{90:95})
radiation  = f->radiation({23:30},:,{20:25},{90:95})
ta_tu_mo   = turbulence + moist + radiation
days = (/"D0-6","D0-3","D0","D0+3"/)


data1 = new((/4,dimsizes(physics&level)/),double)
data1(0,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(dim_avg_n_Wrap(physics, 0), 1), 1)
data1(1,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(dim_avg_n_Wrap(radiation, 0), 1), 1)
data1(2,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(dim_avg_n_Wrap(turbulence, 0), 1), 1)
data1(3,:) = dim_avg_n_Wrap(dim_avg_n_Wrap(dim_avg_n_Wrap(moist, 0), 1), 1)
;exit

plot = new(1,graphic)
dum  = new(1,graphic)
wks  = gsn_open_wks("pdf", path2+"re_to_reviewer_sdiabatic_heating_postmonsoon")

res = True
res@gsnDraw               = False
res@gsnFrame              = False
res@gsnRightString        = "(10N,80E)"
res@gsnStringFontHeightF        =  0.04

res@trXMaxF          =    8
res@trXMinF          =    -4
res@trYReverse       =    True

res@tmYLLabelFontHeightF      = 0.024
res@tmXBLabelFontHeightF      = 0.024

res@xyLineThicknessF =    1.8
res@xyDashPattern    =    0
res@xyMarkLineMode   =    "MarkLines"
res@xyMarkers        =    (/3,4,5,6/)
res@xyLineColors    :=    (/"black","azure4","red","blue"/)

res@lgPerimOn              = False  

polyres                   = True
polyres@gsLineColor       = "black"
polyres@gsLineThicknessF  = 1.5
polyres@gsLineDashPattern = 2


dd  =  (/24,27,30,33/)  
day = 0
res@gsnLeftString = days(day)
plot(day) = gsn_csm_xy(wks,data1(:, :),physics&level,res)
dum(day)  = gsn_add_polyline(wks, plot(day), (/0,0/), (/0,1000/), polyres)


lgres                       = True   
lgres@lgMonoLineThickness   = True
lgres@lgLineThicknessF      = 2.0
lgres@lgPerimOn             = False              
lgres@lgLineLabelStrings    = (/"",""/)          
lgres@lgAutoManage          = False
lgres@lgLineDashSegLenF     = 0.08
lgres@lgLabelFontHeightF    = 0.06   
lgres@lgDashIndexes         = (/0,0/)     

lgres@vpHeightF             = 0.12                  
lgres@vpWidthF              = 0.145    

lgres@lgLineColors          = (/"black"/)
gsn_legend_ndc(wks,1,(/" total"/),0.15,0.15,lgres)
lgres@lgLineColors          = (/"azure4"/)
gsn_legend_ndc(wks,1,(/" radiation"/),0.35,0.15,lgres)
lgres@lgLineColors          = (/"red"/)
gsn_legend_ndc(wks,1,(/"  sensible"/),0.55,0.15,lgres)
lgres@lgLineColors          = (/"blue"/)
gsn_legend_ndc(wks,1,(/" moist"/),0.75,0.15,lgres)

pnlres = True
pnlres@lgPerimOn             = False
pnlres@lgAutoManage          = False
pnlres@lgLineLabelStrings    = (/"",""/)
pnlres@gsnPanelDebug     = True  
;pnlres@gsnPanelMainString = "(10N,80E) vertical heating profile"
gsn_panel(wks,plot,(/1,4/),pnlres)
end