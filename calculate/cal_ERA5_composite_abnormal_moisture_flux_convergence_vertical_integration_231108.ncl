; 2023-11-8
; This script calculate two variables:
;   1. single layer moisture transportation
;   2. vertical integration of the moisture transportation
begin
    ; ------------1. Read the data-------------------------
    file_path = "/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/"
    u_name    = "u_component_of_wind_composite_abnormal.nc"
    v_name    = "v_component_of_wind_composite_abnormal.nc"
    q_name    = "specific_humidity_composite_abnormal.nc"

    file_path_single = "/home/sun/data/ERA5_data_monsoon_onset/composite_ERA5/single/"

    u_file    = addfile(file_path + u_name, "r")
    v_file    = addfile(file_path + v_name, "r")
    q_file    = addfile(file_path + q_name, "r")
    ; Note the surface pressure unit is Pa
    ps_file0  = addfile(file_path_single + "sp_composite_year_early.nc", "r")
    ps_file1  = addfile(file_path_single + "sp_composite_year_late.nc",  "r")

    u_early   = u_file->u_early
    u_late    = u_file->u_late

    v_early   = v_file->v_early
    v_late    = v_file->v_late

    q_early   = q_file->q_early
    q_late    = q_file->q_late

    ps_early  = ps_file0->sp
    ps_late   = ps_file1->sp

    lat       = u_file->lat
    lon       = u_file->lon
    lev       = u_file->lev

    value = 1.e20
    replace_ieeenan(u_early, value, 0)
    replace_ieeenan(u_late, value, 0)
    replace_ieeenan(v_early, value, 0)
    replace_ieeenan(v_late, value, 0)
    u_early@_FillValue = value
    u_late@_FillValue = value
    v_early@_FillValue = value
    v_late@_FillValue = value

    ; =============2. calculate moisture transportation ==============
    mtu_early = u_early * q_early / 9.8
    mtv_early = v_early * q_early / 9.8
    
    mtu_late  = u_late  * q_late / 9.8
    mtv_late  = v_late  * q_late / 9.8

    ; ============= 3. vertical integration moisture flux divergence ==============
    copy_VarCoords(u_early, mtu_early)
    copy_VarCoords(u_early, mtv_early)
    copy_VarCoords(u_early, mtu_late )
    copy_VarCoords(u_early, mtv_late )

    mt_dv_early  =  uv2dv_cfd(mtu_early, mtv_early, lat, lon, 2)
    mt_dv_late   =  uv2dv_cfd(mtu_late,  mtv_late,  lat, lon, 2)

    ; Calculate vertical integration
    pbot = 1000
    ptop = 300
    mt_vint_early  =  new((/61, 181, 360/), double)
    mt_vint_late   =  new((/61, 181, 360/), double)

    do tt = 0, 60
    print(tt)
        do latt = 0, 180
            do lonn = 0, 359
                mt_vint_early(tt, latt, lonn) = vibeta(lev(::-1), mt_dv_early(tt, :, latt, lonn), 1, ps_early(tt, latt, lonn) / 100, pbot, ptop)
                mt_vint_late(tt, latt, lonn)  = vibeta(lev(::-1), mt_dv_late(tt, :, latt, lonn),  1, ps_late(tt, latt, lonn)  / 100, pbot, ptop)
            end do
        end do
    end do

    ;copy_VarCoords(ps_early, mt_vint_early)
    ;copy_VarCoords(ps_early, mt_vint_late)
    ;print(mt_vint_early)

    ; =========== 4. Write to ncfile==================
    out_file    =  "moisture_transportation_and_vertical_integration_composite_abnormal.nc"
    system("rm -rf " + file_path + out_file)
    ;setfileoption("nc","Format","LargeFile")
    setfileoption("nc","Format","NetCDF4")

    fout        =  addfile(file_path + out_file, "c")

    ;filedimdef(fout,"time",-1,True)

    fout->mt_vint_early  =  mt_vint_early
    fout->mt_vint_late   =  mt_vint_late

    fout->mtu_early      =  mtu_early
    fout->mtu_late       =  mtu_late

    fout->mtv_early      =  mtv_early
    fout->mtv_late       =  mtv_late



end