;2021/12/30
;本代码绘制论文version1.2 中的fig3
;内容为绘制4月的印度半岛与中南半岛的垂直加热廓线
;经向10-15平均，经度为75 100

begin
path  =  "/home/sun/qomo-data/"
path2 =  "/home/sun/paint/lunwen/version3.0/"

f1     =  addfile("/home/sun/data/merra2_climate_vars_multi/monthly/Merra_diabatic_physics_maritime.nc","r")

climate = f1->climate_profile
early   = f1->early_profile
late    = f1->late_profile

value = 1.e20
replace_ieeenan(climate   , value, 0)
replace_ieeenan(early   , value, 0)
replace_ieeenan(late , value, 0)
climate@_FillValue      = value
early@_FillValue      = value
late@_FillValue    = value

data1        = new((/3,dimsizes(climate&lev)/),double)
printVarSummary(data1)
data1(0,:) = climate(:, 2) * 86400
data1(1,:) = early(:,   2) * 86400
data1(2,:) = late(:,    2) * 86400

print(data1(0,:) - data1(2,:))
;
;exit

;;------------------------------------------------------
plot = new(1,graphic)
dum  = new(1,graphic)
;wks  = gsn_open_wks("pdf",path2+"fig6_heating_profile_Apr_liuxl")
wks  = gsn_open_wks("pdf","/home/sun/paint/phd/phd_c5_fig13_heating")

res                         =    True
res@gsnDraw                 =    False
res@gsnFrame                =    False
res@gsnRightString          =    " "
res@gsnStringFontHeightF    =     0.04

res@trXMaxF                 =    12
res@trXMinF                 =    -4
res@trYReverse              =    True

res@tmYLLabelFontHeightF    =    0.017
res@tmXBLabelFontHeightF    =    0.017
res@tmXBLabelDeltaF         =    -0.6   ;; ; push the XB labels up from plot
res@tmYLLabelDeltaF         =    -0.6

res@tmXBMajorOutwardLengthF =    0.006
res@tmXBMinorOutwardLengthF =    0.006
res@tmYLMajorOutwardLengthF =    0.006
res@tmYLMinorOutwardLengthF =    0.006


res@tmXTOn                  =    False
res@tmYROn                  =    False

res@xyLineThicknessF        =    3.5
res@xyDashPattern           =    0
res@xyMarkLineMode          =    "MarkLines"
res@xyMarkers               =    (/6,6,6/)
res@xyLineColors            =    (/"black","blue","red"/)

res@lgPerimOn               =    False  

polyres                     =    True
polyres@gsLineColor         =    "black"
polyres@gsLineThicknessF    =    1.5
polyres@gsLineDashPattern   =    2

res@gsnRightString           =    "  "
res@gsnRightStringFontHeightF=    0.026
plot = gsn_csm_xy(wks,data1,climate&lev,res)
;dum  = gsn_add_polyline(wks, plot, (/0,0/), (/200,1000/), polyres)

;************************************************
  txres               = True                     ; text mods desired
  txres@txFontHeightF = 0.03                     ; font smaller. default big

  dum = gsn_add_text(wks,plot,"(c)",-3,250,txres) 

draw(plot)

lgres                       =    True   
lgres@lgMonoLineThickness   =    True
lgres@lgLineThicknessF      =    2.5
lgres@lgPerimOn             =    False              
lgres@lgLineLabelStrings    =    (/"",""/)          
lgres@lgAutoManage          =    False
lgres@lgLineDashSegLenF     =    0.15
lgres@lgLabelFontHeightF    =    0.34   
lgres@lgDashIndexes         =    (/0,0/)     

lgres@vpHeightF             =    0.295                  
lgres@vpWidthF              =    0.24    

;lgres@lgLineColors          = (/"black"/)
;gsn_legend_ndc(wks,1,(/" total"/),0.15,0.15,lgres)
;lgres@lgLineColors          = (/"azure4"/)
;gsn_legend_ndc(wks,1,(/" radiation"/),0.35,0.15,lgres)

;lgres@lgLineColors          = (/"red"/)
;gsn_legend_ndc(wks,1,(/"  sensible"/),0.55,0.15,lgres)
;lgres@lgLineColors          = (/"blue"/)
;gsn_legend_ndc(wks,1,(/" moist"/),0.75,0.15,lgres)
;lgres@lgMonoLineColor       = False
;lgres@lgLineColors          = (/"red"/)
;lgres@lgLabelFontColor      = (/"red"/)
;gsn_legend_ndc(wks,1,(/" sensible heating"/),0.53,0.90,lgres)
;
;lgres@lgLineColors          = (/"blue"/)
;lgres@lgLabelFontColor      = (/"blue"/)
;gsn_legend_ndc(wks,1,(/"   latent  heating"/),0.53,0.86,lgres)
;
;lgres@lgLineColors          = (/"black"/)
;lgres@lgLabelFontColor      = (/"black"/)
;gsn_legend_ndc(wks,1,(/"   total   heating"/),0.53,0.82,lgres)
end

;调图记录
;前20天平均和前30天结果一样