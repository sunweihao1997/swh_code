; load functions and procedures
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
;load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"

begin

; WRF ARW input file (NOTE, your wrfout file does not need
; the .nc, but NCL needs it so make sure to add it in the
; line below)
a = addfile("/home/sun/Build_WRF/WRF-4.4-ARW/run/wrfout_d01_2024-09-04_00:05:00.nc","r")


; Output on screen. Output will be called "plt_Surface1"
type = "png"
wks = gsn_open_wks(type,"plt_Surface1")


; Set basic resources
res = True
res@MainTitle = "REAL-TIME WRF"          ; Give plot a main title
res@Footer = False                              ; Set Footers off
pltres = True                                ; Plotting resources
mpres = True                                      ; Map resources

;---------------------------------------------------------------
times = wrf_user_getvar(a,"times",-1)    ; get times in the file
it = 0                            ; only interested in first time
res@TimeLabel = times(it)            ; keep some time information

;---------------------------------------------------------------
; Get variables

slp = wrf_user_getvar(a,"slp",it)                     ;    Get slp
   wrf_smooth_2d( slp, 3 )                           ; Smooth slp
t2 = wrf_user_getvar(a,"T2",it)                  ; Get T2 (deg K)
   tc2 = t2-273.16                             ; Convert to deg C
   tf2 = 1.8*tc2+32.                           ; Convert to deg F
   tf2@description = "Surface Temperature"
   tf2@units = "F"

u10 = wrf_user_getvar(a,"U10",it)                       ; Get U10
v10 = wrf_user_getvar(a,"V10",it)                       ; Get V10
   u10 = u10*1.94386                           ; Convert to knots
   v10 = v10*1.94386
   u10@units = "kts"
   v10@units = "kts"

;---------------------------------------------------------------


; Plotting options for T
opts = res                                  ; Add basic resources
opts@cnFillOn = True                                ; Shaded plot
opts@ContourParameters = (/ -20., 90., 5./)   ; Contour intervals
opts@gsnSpreadColorEnd = -3
contour_tc = wrf_contour(a,wks,tf2,opts)            ; Create plot
delete(opts)


; Plotting options for SLP
opts = res                                  ; Add basic resources
opts@cnLineColor = "Blue"                        ; Set line color
opts@cnHighLabelsOn = True                           ; Set labels
opts@cnLowLabelsOn = True
opts@ContourParameters = (/ 900.,1100.,4./)   ; Contour intervals
contour_psl = wrf_contour(a,wks,slp,opts)           ; Create plot delete(opts)


; Plotting options for Wind Vectors
opts = res                                  ; Add basic resources
opts@FieldTitle = "Winds"             ; Overwrite the field title
opts@NumVectors = 47                      ; Density of wind barbs
vector = wrf_vector(a,wks,u10,v10,opts)             ; Create plot
delete(opts)


; MAKE PLOTS
plot = wrf_map_overlays(a,wks,  \
      (/contour_tc,contour_psl,vector/),pltres,mpres)

;---------------------------------------------------------------


end