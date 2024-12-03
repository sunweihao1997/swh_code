;2021/12/30
;本代码绘制论文version1.2 中的fig3
;内容为绘制4月的印度半岛与中南半岛的垂直加热廓线
;经向10-15平均，经度为75 100

begin


f1     =  addfile("/home/sun/mydown/MERRA2_diabatic_heating_selection/MERRA2_400.tavgU_3d_tdt_Np.202404.nc4.nc4","r")

;Maritime Continent
moist2       =  dim_avg_n_Wrap( f1->DTDTMST(:,{1000:200},{0:10},{110:130}),     0)  

moist_avg    =  dim_avg_n_Wrap(moist2,  (/1,2/))
moist_avg    = moist_avg * 86400
;print(moist_avg)

;exit
;exit
;;------------------------------------------------------
plot = new(1,graphic)
dum  = new(1,graphic)
;wks  = gsn_open_wks("pdf",path2+"fig6_heating_profile_Apr_liuxl")
wks  = gsn_open_wks("png","lunwen_fig2c_v4.0_heating_profile_Apr_ocn_missing")

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
res@xyMarkerColors          =    (/"maroon4","maroon4","maroon4","green4","green4","green4"/)
res@xyMarkers               =    (/7,7,7,16,16,16/)
res@xyLineColors            =    (/"red","blue","ivory3","red","blue","ivory3"/)

res@lgPerimOn               =    False  

polyres                     =    True
polyres@gsLineColor         =    "black"
polyres@gsLineThicknessF    =    1.5
polyres@gsLineDashPattern   =    2

res@gsnRightString           =    "  "
res@gsnRightStringFontHeightF=    0.026
plot = gsn_csm_xy(wks,moist_avg,turbulence1&level,res)
dum  = gsn_add_polyline(wks, plot, (/0,0/), (/200,1000/), polyres)

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
lgres@lgMonoLineColor       = False
lgres@lgLineColors          = (/"red"/)
lgres@lgLabelFontColor      = (/"red"/)
gsn_legend_ndc(wks,1,(/" sensible heating"/),0.53,0.90,lgres)

lgres@lgLineColors          = (/"blue"/)
lgres@lgLabelFontColor      = (/"blue"/)
gsn_legend_ndc(wks,1,(/"   latent  heating"/),0.53,0.86,lgres)

lgres@lgLineColors          = (/"ivory3"/)
lgres@lgLabelFontColor      = (/"ivory3"/)
gsn_legend_ndc(wks,1,(/"   total   heating"/),0.53,0.82,lgres)
end

;调图记录
;前20天平均和前30天结果一样