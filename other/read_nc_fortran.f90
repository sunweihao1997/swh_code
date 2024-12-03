program read_netcdf
    use netcdf
    implicit none

    integer :: ncid, varid, status
    integer :: ndims, dim1, dim2
    integer, dimension(:), allocatable :: dimids
    real, dimension(:,:), allocatable :: temperature

    ! 使用绝对路径
    character(len=100) :: filename = "/home/sun/data/ERA5_single_sst_tp_May_1980_2014.nc"

    ! 打开NetCDF文件
    status = nf90_open(filename, NF90_NOWRITE, ncid)
    if (status /= nf90_noerr) then
        print *, "Error opening file:", trim(filename)
        stop
    end if

    ! 获取变量ID
    status = nf90_inq_varid(ncid, "temperature", varid)
    print *, status

    ! 获取维度信息
    status = nf90_inq_varndims(ncid, varid, ndims)
    allocate(dimids(ndims))
    status = nf90_inq_vardimid(ncid, varid, dimids)

!    ! 获取各个维度的大小
!    status = nf90_inq_dimlen(ncid, dimids(1), dim1)
!    status = nf90_inq_dimlen(ncid, dimids(2), dim2)
!    allocate(temperature(dim1, dim2))
!
!    ! 读取变量数据
!    status = nf90_get_var(ncid, varid, temperature)
!    if (status /= nf90_noerr) then
!        print *, "Error reading temperature data"
!        stop
!    end if
!
!    ! 打印部分数据以验证
!    print *, "Temperature data:"
!    print *, temperature

    ! 关闭NetCDF文件
    status = nf90_close(ncid)
    if (status /= nf90_noerr) then
        print *, "Error closing file"
    end if

end program read_netcdf
