begin
path  = "/home/sun/data/model_data/climate/b1850_exp/"
f0    = addfile(path+"b1850_control_climate_atmosphere.nc","r")
f1    = addfile(path+"b1850_inch_climate_atmosphere.nc","r")
f2    = addfile(path+"b1850_indian_climate_atmosphere3.nc","r")
;f3    = addfile(path+"b1850_inch_indian_atmosphere.nc","r")
f4    = addfile(path+"b1850_maritime_atmosphere_new020.nc","r")
;------------- April - June  ----------------------------------
t0    = f0->T(90:180,{500:200},:,:) ; control
t1    = f1->T(90:180,{500:200},:,:) ; inch
t2    = f2->T(90:180,{500:200},:,:) ; indian
;t3    = f3->T(90:180,{500:200},:,:) ; inch_indian
t4    = f4->T(90:180,{500:200},:,:) ; maritime
printVarSummary(t0) 

;[time | 91] x [lev | 5] x [lat | 192] x [lon | 288]
;April 1-30  90:119  May 1-31: 120-150    June：151:180  
;-------------------bob-------------------------------------
tbob       = new((/4,91/),"float")
tbob(0,:)  = dim_avg_n_Wrap(t0(:,:,{20},{60:75}),(/1,2/)) - dim_avg_n_Wrap(t0(:,:,{10},{60:75}),(/1,2/))
tbob(1,:)  = dim_avg_n_Wrap(t1(:,:,{20},{60:75}),(/1,2/)) - dim_avg_n_Wrap(t1(:,:,{10},{60:75}),(/1,2/))
tbob(2,:)  = dim_avg_n_Wrap(t2(:,:,{20},{60:75}),(/1,2/)) - dim_avg_n_Wrap(t2(:,:,{10},{60:75}),(/1,2/))
tbob(3,:)  = dim_avg_n_Wrap(t4(:,:,{20},{60:75}),(/1,2/)) - dim_avg_n_Wrap(t4(:,:,{10},{60:75}),(/1,2/))
;tbob(4,:)  = dim_avg_n_Wrap(t4(:,:,{15},{90:100}),(/1,2/)) - dim_avg_n_Wrap(t4(:,:,{5},{90:100}),(/1,2/))
;tbob(4,30:60)  = (0.3*tbob(1,35:60) + 0.7*tbob(4,35:60)) / 2
tbob!1     = "time"
tbob&time  = ispan(1,91,1)

;copy_VarCoords(t0(:,0,0,0),tbob(0,:));[4] x [time | 121]
printVarSummary(tbob)
;print(tbob(2,:))

;1,11,21,     31, 41,51,61,  (62)     71, 81, 91,  
;apr 1，21    May1      31   June1     10  20  30   
;------------------------------------------
wks   = gsn_open_wks ("pdf","/home/sun/paint/phd/phd_c4_v0_mtg_india")                           
resL                        = True 
resL@gsnDraw                = False                   ; don't draw
resL@gsnFrame               = False	
resL@xyLineThicknesses      = (/5.0,5.0,5.0,5.0/)
resL@trYMaxF                =  3.                
resL@trYMinF                = -3. 
resL@tiYAxisString          = "MTG"
resL@tiXAxisString          = "   "
resL@tiMainString           = "  "
resL@gsnRightString         = "BOB"
resL@xyExplicitLegendLabels = (/"CTRL","No_Inch","No_Indian","No_Mari_Continent"/)        
resL@xyLineColors           = (/"black","brown1","tan1","darkorchid2"/) 
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

labels                      = (/"No_maritime","No_Inch","No_India","CTRL"/)
lgres                       = True
lgres@lgLineColors          = (/"darkorchid2","brown1","tan1","black"/)    ; colors for legend lines
lgres@vpWidthF              = 0.28                   ; width of legend (NDC)
lgres@vpHeightF             = 0.16                   ; height of legend (NDC)
lgres@lgPerimOn             = False                 ; turn off perimeter
lgres@lgDashIndexes         = (/0,0,0,0/)              ; dash indexes
lgres@lgLineThicknessF      = resL@xyLineThicknesses
lgres@lgLineLabelStrings    = labels             ; no labels in lines
gsn_legend_ndc(wks,4,labels,0.15,0.8,lgres)

draw(plot)
frame(wks)
end