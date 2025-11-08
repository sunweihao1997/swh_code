#!/usr/bin/env csh
# c-shell script to download selected files from gdex.ucar.edu using Wget

set opts = "-N"

# Check wget version. Set the --no-check-certificate option if wget >= 1.10
set v = `wget -V | grep 'GNU Wget ' | cut -d ' ' -f 3`
set a = `echo $v | cut -d '.' -f 1`
set b = `echo $v | cut -d '.' -f 2`
if ( 100 * $a + $b > 109 ) then
  set cert_opt = "--no-check-certificate"
else
  set cert_opt = ""
endif

set filelist = ( \
  https://osdf-data.gdex.ucar.edu/ncar/gdex/d651056/CESM2-LE/ocn/proc/tseries/month_1/SST/b.e21.BSSP370cmip6.f09_g17.LE2-1251.010.pop.h.SST.202501-203412.nc \
  https://osdf-data.gdex.ucar.edu/ncar/gdex/d651056/CESM2-LE/ocn/proc/tseries/month_1/SST/b.e21.BHISTcmip6.f09_g17.LE2-1081.005.pop.h.SST.193001-193912.nc \
)

while ( $#filelist > 0 )
  set syscmd = "wget $cert_opt $opts $filelist[1]"
  echo "$syscmd ..."
  eval $syscmd
  shift filelist
end
