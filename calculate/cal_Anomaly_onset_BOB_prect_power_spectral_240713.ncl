; 2024-5-15
; This script is for the test purpose to do power spectrum analysis for single file

begin
    f0 = addfile("/home/sun/data/monsoon_onset_anomaly_analysis/process/ERA5_BOB_prect_Apr_May_area_average.nc", "r")


    ;print(f0)
    olr  = f0->tp_series

    printMinMax(olr, 0)
    exit

    value = 1.e20
    replace_ieeenan(olr, value, 0)

    olr@_FillValue = value

    print(olr)


    ;print(prect_ea)


    ;************************************************
    ; set function arguments
    ;************************************************
    d   = 1    ; detrending opt: 0=>remove mean 1=>remove mean + detrend
    sm  = 13   ; smooth: should be at least 3 and odd
    pct = 0.1 ; percent taper: (0.0 <= pct <= 1.0) 0.10 common. 
    ;************************************************

    ps_sa = specx_anal(olr,d,sm,pct)

    ;print(ps_sa)

    ;************************************************
    ; plotting parameters
    ;************************************************
    wks  = gsn_open_wks("pdf","/home/sun/paint/phd/phd_C5_fig1b_v0_area_spectral_power")             ; send graphics to PNG file
    plot = new(1,graphic)                         ; create graphic array 

    r               = True                        ; plot mods desired
    r@gsnDraw       = False                       ; do not draw
    r@gsnFrame      = False                       ; do not advance frame
    r@tiMainString  = "Eastern BOB"                       ; title
    ;r@tiMainPosition= "Left"
    r@tiXAxisString = "Frequency (day)"  ; xaxis
    r@tiYAxisString = "Variance"                  ; yaxis
    r@trXMinF = 5
    r@trXMaxF = 120
    r@xyLineThicknessF = 5
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