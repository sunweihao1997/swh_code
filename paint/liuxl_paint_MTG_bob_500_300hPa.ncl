begin
path  = "/home/sun/data/model_data/climate/b1850_exp/"
f0    = addfile(path+"b1850_control_climate_atmosphere.nc","r")
f1    = addfile(path+"b1850_inch_climate_atmosphere.nc","r")
f2    = addfile(path+"b1850_indian_climate_atmosphere3.nc","r")
f3    = addfile(path+"b1850_inch_indian_atmosphere.nc","r")
;------------- April - June  ----------------------------------
t0    = f0->T(90:180,{500:300},:,:) ; control
t1    = f1->T(90:180,{500:300},:,:) ; inch
t2    = f2->T(90:180,{500:300},:,:) ; indian
t3    = f3->T(90:180,{500:300},:,:) ; inch_indian
printVarSummary(t0) 

;[time | 91] x [lev | 5] x [lat | 192] x [lon | 288]
;April 1-30  90:119  May 1-31: 120-150    June：151:180  
;-------------------bob-------------------------------------
tbob       = new((/4,91/),"float")
print(t1(0,0,{15},{85:95}))
tbob(0,:)  = dim_avg_n_Wrap(t0(:,:,{15},{85:95}),(/1,2/)) - dim_avg_n_Wrap(t0(:,:,{5},{85:95}),(/1,2/))
tbob(1,:)  = dim_avg_n_Wrap(t1(:,:,{15},{85:95}),(/1,2/)) - dim_avg_n_Wrap(t1(:,:,{5},{85:95}),(/1,2/))
tbob(2,:)  = dim_avg_n_Wrap(t2(:,:,{15},{85:95}),(/1,2/)) - dim_avg_n_Wrap(t2(:,:,{5},{85:95}),(/1,2/))
tbob(3,:)  = dim_avg_n_Wrap(t3(:,:,{15},{85:95}),(/1,2/)) - dim_avg_n_Wrap(t3(:,:,{5},{85:95}),(/1,2/))
tbob!1     = "time"
tbob&time  = ispan(1,91,1)

;copy_VarCoords(t0(:,0,0,0),tbob(0,:));[4] x [time | 121]
printVarSummary(tbob)
;print(tbob(2,:))

;1,11,21,     31, 41,51,61,  (62)     71, 81, 91,  
;apr 1，21    May1      31   June1     10  20  30   
;------------------------------------------
wks   = gsn_open_wks ("pdf","/home/sun/paint/phd/phd_c4_liuxl_Ts_bob")                           
resL                        = True 
resL@gsnDraw                = False                   ; don't draw
resL@gsnFrame               = False	
resL@xyLineThicknesses      = (/5.0,5.0,5.0,5.0/)
resL@trYMaxF                =  4.                
resL@trYMinF                = -4. 
resL@tiYAxisString          = "MTG"
resL@tiXAxisString          = "   "
resL@tiMainString           = "  "
resL@gsnRightString         = "BOB"
resL@xyExplicitLegendLabels = (/"CTRL","No_India","No_Inch","No_Indian_Inch"/)        
resL@xyLineColors           = (/"black","brown1","tan1","steelblue2"/) 
resL@xyDashPatterns         = (/0,0,0,0/)
resL@tmXTOn                 = False
resL@trXMinF                =  0 ;90+0
resL@trXMaxF                = 92  ;122+90
resL@tmXBMode               ="Explicit"
resL@tmXBValues             =(/1,11,21, 31,41,51,61,      71,81,91/)
;resL@tmXBLabels             =" " + resL@tmXBValues
resL@tmXBLabels             = (/"Apr01","11","21","May01","11","21","31",   "Jun10","20","30"/)
resL@tmYRMode               = "Automatic"
resL@tmYROn                 = False
resL@vpKeepAspect           = False
resL@vpXF                   = 0.1
resL@vpWidthF               = 0.8    
resL@vpHeightF              = 0.5
resL@gsnYRefLine            =(/0.0/)
resL@gsnYRefLineDashPattern = (/2/)
resL@gsnYRefLineThicknesses = (/3.0/)
plot = gsn_csm_xy(wks,tbob&time,tbob(:,:),resL)

labels                      = (/"No_Indian_Inch","No_Inch","No_India","CTRL"/)
lgres                       = True
lgres@lgLineColors          = (/"steelblue2","tan1","brown1","black"/)    ; colors for legend lines
lgres@vpWidthF              = 0.28                   ; width of legend (NDC)
lgres@vpHeightF             = 0.1                   ; height of legend (NDC)
lgres@lgPerimOn             = False                 ; turn off perimeter
lgres@lgDashIndexes         = (/0,0,0,0/)              ; dash indexes
lgres@lgLineThicknessF      = resL@xyLineThicknesses
lgres@lgLineLabelStrings    = labels             ; no labels in lines
gsn_legend_ndc(wks,4,labels,0.5,0.78,lgres)

draw(plot)
frame(wks)
end