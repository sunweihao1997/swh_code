;2024-6-5
;This script is to plot the meridional_vertical cross-section for some variables, which is to show the environment for the propagating of ISO

begin

; ================= File Information ===============

file0  =  addfile("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_monsoon_onset_climate_early_late_pressure_variables_composite_average_monthly_new.nc", "r")

var0_climate   =  file0->w_climate(:, {200:1000}, :, {80:100})
var0_early     =  file0->w_early(:, {200:1000}, :, {80:100})
var0_late      =  file0->w_late(:, {200:1000}, :, {80:100})

; ==================================================

; calculate the deviation for the given month
mon_num = 2
diff_early     =  100*(dim_avg_n_Wrap(var0_early(mon_num, :, :, :), 2) - dim_avg_n_Wrap(var0_climate(mon_num, :, :, :), 2))
diff_late      =  100*(dim_avg_n_Wrap(var0_late(mon_num, :, :, :), 2)  - dim_avg_n_Wrap(var0_climate(mon_num, :, :, :), 2))

copy_VarCoords(dim_avg_n_Wrap(var0_early(mon_num, :, :, :), 2), diff_early)
copy_VarCoords(dim_avg_n_Wrap(var0_late(mon_num, :, :, :), 2), diff_late)

; plot the picture
wks = gsn_open_wks("pdf","/home/sun/paint/phd/phd_c5_fig67_vertical_meridional_March_Omega_ERA5_BOB_late")
;wks = gsn_open_wks("pdf","/home/sun/paint/monsoon_onset_composite_ERA5/vertical_meridional_April_Omega_ERA5_BOB_early")

res                      = True                 ; plot mods desired

res@cnLevelSelectionMode = "ExplicitLevels"       ; manually select levels
;res@cnMinLevelValF  = -4.0             ; 等值线的最小值
;res@cnMaxLevelValF  = 4.0              ; 等值线的最大值
;res@cnLevelSpacingF = 0.5  
res@cnLevels             = fspan(-3, 3, 11)

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