;2024-6-5
;This script is to plot the meridional_vertical cross-section for some variables, which is to show the environment for the propagating of ISO

begin

; ================= File Information ===============

file0  =  addfile("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_monsoon_onset_climate_early_late_pressure_variables_composite_average_monthly_new.nc", "r")

varu_climate   =  file0->u_climate(:, {200:1000}, :, {80:100})
varu_early     =  file0->u_early(:, {200:1000}, :, {80:100})
varu_late      =  file0->u_late(:, {200:1000}, :, {80:100})

varv_climate   =  file0->v_climate(:, {200:1000}, :, {80:100})
varv_early     =  file0->v_early(:, {200:1000}, :, {80:100})
varv_late      =  file0->v_late(:, {200:1000}, :, {80:100})

value = 1.e20
replace_ieeenan(varu_climate, value, 0)
replace_ieeenan(varu_early, value, 0)
replace_ieeenan(varu_late, value, 0)
replace_ieeenan(varv_climate, value, 0)
replace_ieeenan(varv_early, value, 0)
replace_ieeenan(varv_late, value, 0)
varu_climate@_FillValue = value
varu_early@_FillValue = value
varu_late@_FillValue = value
varv_climate@_FillValue = value
varv_early@_FillValue = value
varv_late@_FillValue = value

lat            =  file0->latitude
lon            =  file0->longitude

;print(lat)

div_climate    =  uv2vr_cfd(varu_climate, varv_climate, varv_climate&latitude, varv_climate&longitude, 1)
div_early      =  uv2vr_cfd(varu_early,   varv_early,   varv_climate&latitude, varv_climate&longitude, 1)
div_late       =  uv2vr_cfd(varu_late,    varv_late,    varv_climate&latitude, varv_climate&longitude, 1)

copy_VarCoords(varu_climate, div_climate)
copy_VarCoords(varu_climate, div_early)
copy_VarCoords(varu_climate, div_late)

; ==================================================

; calculate the deviation for the given month
mon_num = 2
diff_early     =  1000000*(dim_avg_n_Wrap(div_early(mon_num, :, :, :), 2) - dim_avg_n_Wrap(div_climate(mon_num, :, :, :), 2))
diff_late      =  1000000*(dim_avg_n_Wrap(div_late(mon_num, :, :, :), 2)  - dim_avg_n_Wrap(div_climate(mon_num, :, :, :), 2))

;print(dim_avg(diff_early))

copy_VarCoords(dim_avg_n_Wrap(varu_climate(mon_num, :, :, :), 2), diff_early)
copy_VarCoords(dim_avg_n_Wrap(varu_climate(mon_num, :, :, :), 2), diff_late)

; plot the picture
;wks = gsn_open_wks("pdf","/home/sun/paint/monsoon_onset_composite_ERA5/vertical_meridional_April_vor_ERA5_BOB_early")
wks = gsn_open_wks("pdf","/home/sun/paint/phd/vertical_meridional_March_vor_ERA5_BOB_late")

res                      = True                 ; plot mods desired

res@cnLevelSelectionMode = "ExplicitLevels"       ; manually select levels
;res@cnMinLevelValF  = -4.0             ; 等值线的最小值
;res@cnMaxLevelValF  = 4.0              ; 等值线的最大值
;res@cnLevelSpacingF = 0.5  
res@cnLevels             = fspan(-5, 5, 11)

res@trXMinF = -20                 ; 设置横坐标最小值
res@trXMaxF = 20  
res@cnLineLabelsOn       = False                 ; turn on line labels
res@cnFillOn             = True                 ; turn on color fill
res@cnFillPalette        = "cmocean_balance"             ; choose colormap

plot = gsn_csm_pres_hgt(wks,diff_late, res)         ; create plot    

;  print(plot)
;  print(plot@contour)
;  getvalues plot@contour
;    "tmXBValues" : values
;    "tmXBLabels" : labels
;  end getvalues
;  print("tmXBValues / Labels = " + values + " /" + labels)
;

end