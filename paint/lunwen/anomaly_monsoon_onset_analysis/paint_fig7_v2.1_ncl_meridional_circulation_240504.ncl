;2024-4-23
;This function is to plot the zonal circulation of the regressioned circulation to the OLR index
;use ncl other than py is due to the level axis is always wrong
begin
    f0 = addfile("/home/sun/data/ERA5_data_monsoon_onset/regression/ERA5_regression_LSTC-remove-OLR_meridional_tropical_circulation.nc", "r")

    u  = f0->rc_v
    w  = f0->rc_w

    level = f0->level
    level@units = "hpa"
    ;printVarSummary(level)

    wAve = avg(w)
    uAve = avg(u)

    scale = fabs(uAve/wAve)
    ;print(scale)
    wscale = w * 600

    copy_VarCoords(w, wscale)

    u&level = level
    wscale&level=level

  
    wks   = gsn_open_wks ("png", "h_lat" )          ; send graphics to PNG file
  
    res                      = True                 ; plot mods desired
  

    res@cnLineLabelsOn       = True                 ; turn on line labels
    res@cnFillOn             = True                 ; turn on color fill
    res@cnFillPalette        = "BlWhRe"             ; choose colormap

    res@trXMinF = -15     ; X轴的最小值
    res@trXMaxF = 15   ; X轴的最大值
    ;res@tiYAxisString        = u&lev@long_name + " (" + u&lev@units + ")"   
  
  ; Note: u is already on pressure levels. If this were model data, it
  ; would be necessary to interpolate from the hybrid coordinates to 
  ; pressure levels.
    
    plot  = gsn_csm_pres_hgt(wks, u, res )   ; place holder 

end