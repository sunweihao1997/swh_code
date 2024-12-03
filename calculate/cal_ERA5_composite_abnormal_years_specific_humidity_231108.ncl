; 2023-11-8
; This script use ncl function to calculate specific humidity in the composite analysis array
begin
    file_path = "/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/"
    t_file    = "temperature_composite_abnormal.nc"
    rh_file   = "relative_humidity_composite_abnormal.nc"

    ; ----------- read file -------------
    ft        = addfile(file_path + t_file,  "r")
    frh       = addfile(file_path + rh_file, "r")

    t_early   = ft->t_early
    t_late    = ft->t_late

    rh_early  = frh->r_early
    rh_late   = frh->r_late

    value = 1.e20
    replace_ieeenan(t_early, value, 0)
    replace_ieeenan(t_late, value, 0)
    replace_ieeenan(rh_early, value, 0)
    replace_ieeenan(rh_late, value, 0)
    t_early@_FillValue = value
    t_late@_FillValue = value
    rh_early@_FillValue = value
    rh_late@_FillValue = value

    p         = ft->lev
    p_conform = conform(t_early, p, 1)

    ; ---------- Calculate the specific humidity ------------
    sp_early  = mixhum_ptrh(p_conform, t_early, rh_early, -2)
    sp_late   = mixhum_ptrh(p_conform, t_late,  rh_late,  -2)

    copy_VarCoords(t_early, sp_early)
    copy_VarCoords(t_early, sp_late)
    ; =========== 4. Write to ncfile==================
    out_file    =  "specific_humidity_composite_abnormal.nc"
    system("rm -rf " + file_path + out_file)
    ;setfileoption("nc","Format","LargeFile")
    setfileoption("nc","Format","NetCDF4")

    fout        =  addfile(file_path + out_file, "c")

    ;filedimdef(fout,"time",-1,True)

    fout->q_early  =  sp_early
    fout->q_late   =  sp_late



end