;2023-10-23
; This script is to use ncl function to calculate the specific humidity using T and RH
begin
    path_t_btal  =  "/exports/csce/datastore/geos/users/s2618078/data/model_data/T/BTAL/T/"
    path_t_neu   =  "/exports/csce/datastore/geos/users/s2618078/data/model_data/T/BTALnEU/T/"
    path_rh_btal =  "/exports/csce/datastore/geos/users/s2618078/data/model_data/RELHUM/BTAL/RELHUM/"
    path_rh_neu  =  "/exports/csce/datastore/geos/users/s2618078/data/model_data/RELHUM/BTALnEU/RELHUM/"

    do i = 1, 8
        ; =========== 1. Read T and RH data ==============
        file_t_btal   =  path_t_btal + "BTAL_T_1850_150years_member_" + i + ".nc"
        f0            =  addfile(file_t_btal, "r")
        file_t_eu     =  path_t_neu  + "BTALnEU_T_1850_150years_member_" + i + ".nc"
        f1            =  addfile(file_t_eu, "r")

        file_rh_btal  =  path_rh_btal + "BTAL_RELHUM_1850_150years_member_" + i + ".nc"
        f2            =  addfile(file_rh_btal, "r")
        file_rh_eu    =  path_rh_neu  + "BTALnEU_RELHUM_1850_150years_member_" + i + ".nc"
        f3            =  addfile(file_rh_eu, "r")

        ; =========== 2. Deal with P =====================
        p0            =  f0->lev
        t0            =  f0->T
        p1            =  conform(t0, p0, 1)
        
        ; =========== 3. Calculate =======================
        btal_t        =  f0->T
        noeu_t        =  f1->T
        btal_rh       =  f2->RELHUM
        noeu_rh       =  f3->RELHUM

        btal_sp       =  mixhum_ptrh(p1, btal_t, btal_rh, -2)
        noeu_sp       =  mixhum_ptrh(p1, noeu_t, noeu_rh, -2)

        ; =========== 4. Write to ncfile==================
        out_path      =  "/exports/csce/datastore/geos/groups/aerosol/sabineCESM/inp_sun/Specific_humidity/"
        out_file_1    =  "BTAL_Q_1850_150years_member_" + i + ".nc"
        out_file_2    =  "BTALnEU_Q_1850_150years_member_" + i + ".nc"

        fout_1        =  addfile(out_path + out_file_1, "c")
        fout_2        =  addfile(out_path + out_file_2, "c")

        filedimdef(fout_1,"time",-1,True)
        filedimdef(fout_2,"time",-1,True)

        fout_1->Q     =  btal_sp
        fout_2->Q     =  noeu_sp

        print("Finish LOOP " + i)
        ;print(file_t)
    end do

end