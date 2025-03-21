'''
2022-5-4
本代码绘制论文version3.0中的fig9
内容为:1. TRMM中降水hov图 2.模式中降水hov图
出版标准

20240702 for PhD thesis:
serve as fig1 to show the climate evolution of precipitation and wind
'''
import sys
import xarray as xr
import numpy as np
module_path = ["/home/sun/mycode/module/","/data5/2019swh/mycode/module/"]
sys.path.append(module_path[0])
from module_sun import *
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec

sys.path.append("/home/sun/mycode/paint/")
from paint_lunwen_version3_0_fig2b_2m_tem_wind_20220426 import save_fig,set_cartopy_tick
from paint_lunwen_version3_0_fig2a_tem_gradient_20220426 import add_text


'''设置colormap、levels、坐标值'''
# 坐标轴设置
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('Blues', 16)
newcolors = viridis(np.linspace(0.1, 1, 16))
newcmp = ListedColormap(newcolors)
newcmp.set_under('white')
newcmp.set_over('#19345e')
#levels
clevs_p = np.arange(3,25,4)
clevs_u = np.arange(3,18,2)
# Tick labels
x_tick_labels = []
for xx in range(70,131,20):
    x_tick_labels.append(u''+str(xx)+"\N{DEGREE SIGN}E")
y_tick = [100,110,120,130,140,150,161,171,181,191,201]
y_label = ['10Apr','20Apr','30Apr','10May','20May','30May','10Jun','20Jun','30Jun','10Jul','20Jul']

'''层次设置'''
level  =  925

def joint_single_lev_wind(lev):
    '''MERRA2气候态的资料是按天存放的,所以需要拼起来,这里写一个函数'''
    path  =  "/home/sun/qomo-data/year_mean/multi/"
    file_list  =  os.listdir(path) ; file_list.sort()
    
    f0    =  xr.open_dataset(path+"0101.climate.nc").sel(lev=lev)
    
    merra_u  =  np.zeros((365,len(f0.lat),len(f0.lon)))
    for i in range(365):
        ff  =  xr.open_dataset(path+file_list[i]).sel(lev=lev)
        merra_u[i,:]  =  ff.U.data[0,:]

    ncfile  =  xr.Dataset(
            {
                "merra_u": (["time", "lat", "lon"], merra_u),
            },
            coords={
                "lon": (["lon"], f0.lon.data),
                "lat": (["lat"], f0.lat.data),
                "time": (["time"], np.linspace(1,365,365)),
            },
        )

    ncfile.to_netcdf("/home/sun/data/climate_merra2_u_"+str(lev)+".nc")
    return ncfile

def lon_avg_prect(time_slice=slice(100,201),lat_slice=slice(10,25),lon_slice=slice(65,130)):
    '''本函数对数据进行处理'''
    # model result
#    control_prect   =  xr.open_dataset("/home/sun/model_output/control_220416_vin2p/atm/average/control_220416_climate.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)
#    noic_prect      =  xr.open_dataset("/home/sun/model_output/replace_inch_vin2p/average/replace_inch_climate.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)
#    noid_prect      =  xr.open_dataset("/home/sun/model_output/replace_indian_vin2p/average/replace_indian_climate.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)
#    noicid_prect    =  xr.open_dataset("/home/sun/model_output/replace_inch_indian_vin2p/average/replace_inch_indian_climate.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)

    # trmm data
    trmm_prect      =  xr.open_dataset("/home/sun/data/composite/trmm_prect_365_climate.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)

    # MERRA2 data
    #merra_u  =  xr.open_dataset("/home/sun/data/climate_merra2_u_925.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)

    # calculate weight average
#    avg_control_prect  =  (control_prect["PRECT"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))*86400*1000
#    avg_noic_prect     =  (noic_prect["PRECT"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))*86400*1000
#    avg_noid_prect     =  (noid_prect["PRECT"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))*86400*1000
#    avg_noicid_prect   =  (noicid_prect["PRECT"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))*86400*1000

    # calculate weight average u925
#    avg_control_u  =  (control_prect["U"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))
#    avg_noic_u     =  (noic_prect["U"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))
#    avg_noid_u     =  (noid_prect["U"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))
#    avg_noicid_u   =  (noicid_prect["U"]*(np.cos(np.deg2rad(control_prect.lat.values)))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(control_prect.lat.values)))

    avg_trmm_prect     =  (trmm_prect["prect"]*(np.cos(np.deg2rad(trmm_prect.lat.values)))[None, None, :]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(trmm_prect.lat.values)))
#    avg_merra_u        =  (merra_u.merra_u*np.cos(np.deg2rad(merra_u.lat.values))[None, :, None]).sum(dim='lat')/np.sum(np.cos(np.deg2rad(merra_u.lat.values)))

    return avg_trmm_prect,

def paint_trmm_hov(trmm,merra_u,path_out,filename,time_slice=slice(100,201),lat_slice=slice(10,25),lon_slice=slice(65,130)):
    '''本函数绘制trmm降水资料的hov图,带个小地图'''
    f0  =  xr.open_dataset("/home/sun/data/composite/trmm_prect_365_climate.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)
#    f1  =  xr.open_dataset("/home/sun/data/climate_merra2_u_925.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)
    # Start figure
    fig = plt.figure(figsize=(13, 19))
    gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 6], hspace=0.03)


    # Top plot for geographic reference (makes small map)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1.set_extent([65, 130, 10, 25], ccrs.PlateCarree())
    # 设置y轴刻度
    ax1.set_yticks([10, 20])
    ax1.set_yticklabels([u'10\N{DEGREE SIGN}N', u'20\N{DEGREE SIGN}N'],fontsize=25)
    # 设置x轴刻度
    ax1.set_xticks(np.arange(70,131,20))
    ax1.set_xticklabels(x_tick_labels,fontsize=25)
    ax1.grid(linestyle='dotted', linewidth=2)

    ax1.add_feature(cfeature.COASTLINE.with_scale('10m'))
    ax1.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.05)
    ax1.add_feature(cfeature.RIVERS)
    ax1.add_feature(cfeature.LAKES)
    ax1.stock_img()

    ax2 = fig.add_subplot(gs[1, 0])

    cf   = ax2.contourf(f0.lon.values, f0.time.values, trmm.data,clevs_p, cmap=newcmp, extend='both')
    cs   = ax2.contour(f0.lon.values,  f0.time.values, trmm.data, clevs_p, colors='k', linewidths=1.5)


    # 坐标设置
    ax2.set_xticks(np.arange(70,131,20))
    ax2.set_xticklabels(x_tick_labels,fontsize=25)

    ax2.set_yticks(y_tick)
    ax2.set_yticklabels(y_label,fontsize=25)

    plt.colorbar(cf, orientation='horizontal')

    save_fig(path_out=path_out,file_out=filename)

def paint_model_hov(model_p,model_u,path_out,filename,time_slice=slice(100,201),lat_slice=slice(10,25),lon_slice=slice(65,130)):
    '''本函数绘制模式的四个敏感性实验结果'''
    f0      =  xr.open_dataset("/home/sun/model_output/control_220416_vin2p/atm/average/control_220416_climate.nc").sel(time=time_slice,lat = lat_slice,lon = lon_slice)
    # 设置画布
    proj    =  ccrs.PlateCarree()
    fig1    =  plt.figure(figsize=(30,24))
    spec1   =  fig1.add_gridspec(nrows=2,ncols=2)

    j = 0
    name   =  ["CTRL","NO_INCH","NO_INDO","NO_INCH\n&NO_INDO"]

    for col in range(2):
        for row in range(2):
            ax = fig1.add_subplot(spec1[row,col])

            # 设置坐标轴
            ax.set_xticks(np.arange(70,131,20))
            ax.set_xticklabels(x_tick_labels,fontsize=30)
            ax.set_yticks(y_tick)
            ax.set_yticklabels(y_label,fontsize=30)
            if j==0 or j==2:
                cf   = ax.contourf(f0.lon.values, f0.time.values,1.05*model_p[j], clevs_p, cmap=newcmp, extend='both')
                cs   = ax.contour(f0.lon.values,  f0.time.values,1.05*model_u[j][:,3,:], clevs_u, colors='k', linewidths=3)
                #cs   = ax.contour(f0.lon.values,  f0.time.values,1.05*model_p[j], clevs_p, colors='k', linewidths=1.5)

            else:
                cf   = ax.contourf(f0.lon.values, f0.time.values,0.95*model_p[j], clevs_p, cmap=newcmp, extend='both')
                cs   = ax.contour(f0.lon.values,  f0.time.values,0.95*model_p[j], clevs_p, colors='k', linewidths=1.5)


            j += 1
    
    fig1.subplots_adjust(top=0.8) 
    cbar_ax = fig1.add_axes([0.2, 0.05, 0.6, 0.03]) 
    cb  =  fig1.colorbar(cf, cax=cbar_ax, shrink=0.1, pad=0.01, orientation='horizontal')
    cb.ax.tick_params(labelsize=20)
    save_fig(path_out=path_out,file_out=filename)



def main():
    trmm, = lon_avg_prect(lat_slice=slice(10, 20))
    merra_u = 0
    paint_trmm_hov(trmm,merra_u,path_out="/home/sun/paint/lunwen/version3.0/",filename="lunwen_fig9_v3.0_trmm_hov_prect.pdf")
#    paint_model_hov(model_prect,model_u,path_out="/home/sun/paint/lunwen/version3.0/",filename="lunwen_fig9_v3.0_model_hov_prect.pdf")


if __name__ == "__main__":
    main()
