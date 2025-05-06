def set_cartopy_tick(ax, extent, xticks, yticks, nx=0, ny=0,
    xformatter=None, yformatter=None,labelsize=20):
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs
    # 本函数设置地图上的刻度 + 地图的范围
    proj = ccrs.PlateCarree()
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    # 设置次刻度.
    xlocator = mticker.AutoMinorLocator(nx + 1)
    ylocator = mticker.AutoMinorLocator(ny + 1)
    ax.xaxis.set_minor_locator(xlocator)
    ax.yaxis.set_minor_locator(ylocator)

    # 设置Formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    # 设置axi label_size，这里默认为两个轴
    ax.tick_params(axis='both',labelsize=labelsize)

    # 在最后调用set_extent,防止刻度拓宽显示范围.
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=proj)

def create_ncl_colormap(file,bin):
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    rgb  =  []
    with open(file,"r") as f:
        for line in f.readlines():
            line = line.strip('\n')
            line1  =  line.split()
            rgb.append(tuple(np.array(line.split()).astype(float))/max(tuple(np.array(line.split()).astype(float))))
    cmap = LinearSegmentedColormap.from_list('newcmp', rgb, N=bin)

def cal_xydistance(lat,lon):
    from geopy.distance import distance
    disy = np.array([])
    disx = np.array([])
    for i in range(0, (lat.shape[0]-1)):
        disy = np.append(disy, distance((lat[i], 0), (lat[i + 1], 0)).m)

    for i in range(0, lat.shape[0]):
        disx = np.append(disx, distance((lat[i], lon[0]), (lat[i], lon[1])).m)

    location = np.array([0])
    for dddd in range(0, (lat.shape[0]-1)):
        location = np.append(location, np.sum(disy[:dddd + 1]))

    return disy,disx,location

# =========== Paint =======================
import sys
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib as mpl

def colormap_from_list_color(list):
    # 本函数读取颜色列表然后制作出来colormap
    return LinearSegmentedColormap.from_list('chaos',list)

def add_vector_legend(ax,q,location=(0.825, 0),length=0.175,wide=0.2,fc='white',ec='k',lw=0.5,order=1,quiver_x=0.915,quiver_y=0.125,speed=10,fontsize=18):
    '''
    句柄 矢量 位置 图例框长宽 表面颜色 边框颜色  参考箭头的位置 参考箭头大小 参考label字体大小
    '''
    rect = mpl.patches.Rectangle((location[0], location[1]), length, wide, transform=ax.transAxes,    # 这个能辟出来一块区域，第一个参数是最左下角点的坐标，后面是矩形的长和宽
                            fc=fc, ec=ec, lw=lw, zorder=order
                            )
    ax.add_patch(rect)

    ax.quiverkey(q, X=quiver_x, Y=quiver_y, U=speed,
                    label=f'{speed} m/s', labelpos='S', labelsep=0.1,fontproperties={'size':fontsize})

def set_cartopy_tick(ax, extent, xticks, yticks, nx=0, ny=0,
    xformatter=None, yformatter=None,labelsize=20):
    import matplotlib.ticker as mticker
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    # 本函数设置地图上的刻度 + 地图的范围
    proj = ccrs.PlateCarree()
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    # 设置次刻度.
    xlocator = mticker.AutoMinorLocator(nx + 1)
    ylocator = mticker.AutoMinorLocator(ny + 1)
    ax.xaxis.set_minor_locator(xlocator)
    ax.yaxis.set_minor_locator(ylocator)

    # 设置Formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    # 设置axi label_size，这里默认为两个轴
    ax.tick_params(axis='both',labelsize=labelsize)

    # 在最后调用set_extent,防止刻度拓宽显示范围.
    if extent is None:
        ax.set_global()
    else:
        ax.set_extent(extent, crs=proj)

def check_path(path, create):
    '''This function check whether the path exists, and if create parameter is 1, create it when it not exist'''
    import os

    if os.path.exists(path):
        print('{} is already created'.format(path))
    elif create == 1:
        os.system('mkdir -p ' + path)
        print('{} Not exits, has created'.format(path))
    else:
        print('{} not exits'.format(path))

def add_vector_legend(ax,q,location=(0.825, 0),length=0.175,wide=0.2,fc='white',ec='k',lw=0.5,order=1,quiver_x=0.915,quiver_y=0.125,speed=10,fontsize=18):
    '''
    句柄 矢量 位置 图例框长宽 表面颜色 边框颜色  参考箭头的位置 参考箭头大小 参考label字体大小
    '''
    rect = mpl.patches.Rectangle((location[0], location[1]), length, wide, transform=ax.transAxes,    # 这个能辟出来一块区域，第一个参数是最左下角点的坐标，后面是矩形的长和宽
                            fc=fc, ec=ec, lw=lw, zorder=order
                            )
    ax.add_patch(rect)

    ax.quiverkey(q, X=quiver_x, Y=quiver_y, U=speed,
                    label=f'{speed} m/s', labelpos='S', labelsep=0.1,fontproperties={'size':fontsize})

def cal_xydistance(lat,lon):
    from geopy.distance import distance
    disy = np.array([])
    disx = np.array([])
    for i in range(0, (lat.shape[0]-1)):
        disy = np.append(disy, distance((lat[i], 0), (lat[i + 1], 0)).m)

    for i in range(0, lat.shape[0]):
        disx = np.append(disx, distance((lat[i], lon[0]), (lat[i], lon[1])).m)

    location = np.array([0])
    for dddd in range(0, (lat.shape[0]-1)):
        location = np.append(location, np.sum(disy[:dddd + 1]))

    return disy,disx,location