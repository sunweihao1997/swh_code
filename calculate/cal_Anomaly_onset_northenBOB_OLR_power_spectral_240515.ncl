; 2024-5-15
; This script is for the test purpose to do power spectrum analysis for single file

begin
    f0 = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_OLR_Apr_May_normalized.nc", "r")

    ;print(f0)
    olr  = f0->BOB_olr_anomaly

    value = 1.e20
    replace_ieeenan(olr, value, 0)

    olr@_FillValue = value

    print(olr)


    ;print(prect_ea)


    ;************************************************
    ; set function arguments
    ;************************************************
    d   = 1    ; detrending opt: 0=>remove mean 1=>remove mean + detrend
    sm  = 9   ; smooth: should be at least 3 and odd
    pct = 0.1 ; percent taper: (0.0 <= pct <= 1.0) 0.10 common. 
    ;************************************************

    ps_sa = specx_anal(olr,d,sm,pct)

    ;print(ps_sa)

    ;************************************************
    ; plotting parameters
    ;************************************************
    wks  = gsn_open_wks("png","spectral_test")             ; send graphics to PNG file
    plot = new(1,graphic)                         ; create graphic array 

    r               = True                        ; plot mods desired
    r@gsnDraw       = False                       ; do not draw
    r@gsnFrame      = False                       ; do not advance frame
    r@tiMainString  = "South Asia"                       ; title
    ;r@tiMainPosition= "Left"
    r@tiXAxisString = "Frequency (day)"  ; xaxis
    r@tiYAxisString = "Variance"                  ; yaxis
    r@trXMinF = 5
    r@trXMaxF = 120
    ;************************************************
    ; first plot
    ;************************************************
    splt = specx_ci(ps_sa, 0.05, 0.95)
    ;print(splt(0, :))
    f = ps_sa@frq 
    p   = 1/f                                    ; *highly non-linear*
    p!0 = "f"
    p&f = f
    ;ip= ind(p.le.120)
    plot(0)=gsn_csm_xy(wks, p, splt, r)  ; create plot 

    ;***********************************************
    res_P             = True                        ; panel mods desired
    res_P@gsnMaximize = True                        ; blow up plot
    gsn_panel(wks,plot,(/1,1/),res_P)               ; create panel plots

end