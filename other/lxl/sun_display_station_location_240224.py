'''
2024-2-4
The goal of this script is to display the location of the observation station near the port
'''

import os
import sys
import pandas as pd

sys.path.append('/home/sun/mycode/other/lxl')
from lxl_port_data_selection_each_240130 import hanzi_to_pinyin

# Directory
data_path0 = '/home/sun/data/liuxl_sailing/other_data/' # This path includes xlsx which have the lat/lon information of the 15 stations near the one port

geography_list = os.listdir(data_path0)
#print(geography_list[28])

def read_latlon_information(excel_file_name,):
    '''
        The goal of this function is :
        1. Return the lat/lon array for every station
        2. Return the distance for every station
    '''
    # 1. Get the pinyin for the port name
    name_split  = excel_file_name.split(" ")

    port_pinyin = hanzi_to_pinyin(name_split[3])

    print(f'Now it is dealing with the Port {port_pinyin}, this function will return the lat/lon/distance information for each station')

    # 2. Read the excel file
    df          = pd.read_excel(data_path0 + excel_file_name)

    column_name = df.columns[-1] # The last column, corresponding to the distance to the port 

    #print(column_name)
    sorted_df   = df.sort_values(by=column_name) # After sort, it arange from nearest to the furthest

#    print(sorted_df)
    # 3. Get the relevant information
#    print(sorted_df.iloc[0]['lat'])
    geography_information = []
    for i in range(15):
#        print([sorted_df.iloc[i]['lat'], sorted_df.iloc[i]['lon'], sorted_df.iloc[i][column_name]])
        geography_information.append([sorted_df.iloc[i]['lat'], sorted_df.iloc[i]['lon'], sorted_df.iloc[i][column_name]])

#    print(geography_information)
    return geography_information

def read_latlon_information_for_port(excel_file_name,):
    '''
        The goal of the function is to :
        Read the lon/lat information for the port
    '''

    # 1. Get the name of the port
    name_split  = excel_file_name.split(" ")
    port_name   = name_split[3]

    # 2. Compare with the harbour file
    df          = pd.read_excel('/home/sun/data/liuxl_sailing/' + 'harbour.xlsx')

    matching_rows = df.index[df[df.columns[4]] == port_name].tolist() # This command could find which row in the harbour excel corresponds to the port name

    # 3. return the lat/lon for the harbour
    return df.iloc[matching_rows]['lat'].values[0], df.iloc[matching_rows]['lon'].values[0]
#    print(df)
#    print(matching_rows)
#    print(df.iloc[matching_rows])

def plot_harbour_station_map(hlat, hlon, stations, title):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.patches import Circle
    import matplotlib.patches as mpatches

    # 创建一个地图画布，设置投影方式为PlateCarree（这是一种常见的地图投影）
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([hlon - 3.5, hlon + 3.5, hlat - 3.5, hlat + 3.5])  # 设置视图范围为中国的经纬度范围

    # 添加地图特征
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    ax.plot(hlon, hlat, 'o', color='red', markersize=7, transform=ccrs.PlateCarree())

    central_lon, central_lat = hlon, hlat
    radius_deg = 50 / 111  # 大约每度111km
    #circle = Circle((central_lon, central_lat), radius_deg, edgecolor='green', alpha=0.5, transform=ccrs.PlateCarree(), )
    ax.add_patch(mpatches.Circle(xy=[central_lon, central_lat], radius=150 / 111, edgecolor='grey', facecolor='none', alpha=0.85, transform=ccrs.PlateCarree(), zorder=30))
    ax.add_patch(mpatches.Circle(xy=[central_lon, central_lat], radius=100 / 111, edgecolor='black', facecolor='none', alpha=0.85, transform=ccrs.PlateCarree(), zorder=30))
    ax.add_patch(mpatches.Circle(xy=[central_lon, central_lat], radius=50 / 111, edgecolor='brown',  facecolor='none', alpha=0.85, transform=ccrs.PlateCarree(), zorder=30))
    ax.add_patch(mpatches.Circle(xy=[central_lon, central_lat], radius=25 / 111, edgecolor='green',  facecolor='none', alpha=0.85, transform=ccrs.PlateCarree(), zorder=30))
    ax.add_patch(mpatches.Circle(xy=[central_lon, central_lat], radius=10 / 111, edgecolor='red',    facecolor='none', alpha=0.85, transform=ccrs.PlateCarree(), zorder=30))

    for i in range(15):
        ax.plot(stations[i][1], stations[i][0], '^', color='blue', markersize=11.5, transform=ccrs.PlateCarree())

    print('complete')
    plt.savefig('Harbour and Stations Map.png')



def main():
    station_geography = read_latlon_information(geography_list[28])

    hlat, hlon = read_latlon_information_for_port(geography_list[28])
#    print(harbour_geography)
    print(hlat) ; print(hlon)

    plot_harbour_station_map(hlat, hlon, station_geography, 0)

if __name__ == '__main__':
    main()