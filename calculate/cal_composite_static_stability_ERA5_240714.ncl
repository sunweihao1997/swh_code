;2024-7-14
;This script is to calculate the composite potential temp based on ERA5
begin

    src_path = "/home/sun/data/monsoon_onset_anomaly_analysis/ERA5_data_monsoon_onset/composite_ERA5/"

    f_rh     = addfile(src_path + "relative_humidity_composite_abnormal.nc", "r")
    f_t      = addfile(src_path + "temperature_composite_abnormal.nc", "r")

    t_early  = f_t->t_early
    t_late   = f_t->t_late

    r_early  = f_rh->r_early
    r_late   = f_rh->r_late

    p        = f_rh->lev
    pcon     = conform(t_late, p, 1)

    ; Replace the NAN value
    if (any(isnan_ieee(t_early))) then
        print("Include NAN!")
        value = 1.e20
        replace_ieeenan(t_early   , value, 0)
        replace_ieeenan(t_late   , value, 0)
        replace_ieeenan(r_early , value, 0)
        replace_ieeenan(r_late  , value, 0)
        t_early@_FillValue      = value
        t_late@_FillValue      = value
        r_early@_FillValue    = value
        r_late@_FillValue     = value
    end if

    ;;;;;;;;;;;; Calculate the tlcl ;;;;;;;;;;;;;;
    tlcl_early  =  static_stability(p*100, t_early, 1, 0)
    tlcl_late   =  static_stability(p*100, t_late , 1, 0)

    fout = addfile(src_path+"static_stability_composite_abnormal.nc","c")

    ;filedimdef(fout,"time",-1,True)
    fout->time           =  tofloat(tlcl_early&time) 
    fout->theta_e_early  =  tlcl_early
    fout->theta_e_late   =  tlcl_late
 
    fileatt       =  True
    fileatt@time  =  "2024-7-14"
    fileatt@script=  "cal_composite_static_stability_ERA5_240714.ncl mycode_copy ubuntu"
    ;fileatt@description  =  "Using composite temperature to calculate the equivalent PT for ERA5"  

end