'''
2025-12-18
This script calculates the count how many typhoon locates in corresponding marine zone
Relevant tutor link:  https://gemini.google.com/app/119a86e88d9c3d57?is_sa=1&is_sa=1&android-min-version=301356232&ios-min-version=322.0&campaign_id=bkws&utm_source=sem&utm_source=google&utm_medium=paid-media&utm_medium=cpc&utm_campaign=bkws&utm_campaign=2024enIN_gemfeb&pt=9008&mt=8&ct=p-growth-sem-bkws&gclsrc=aw.ds&gad_source=1&gad_campaignid=20357620749&gbraid=0AAAAApk5Bhm6Ffc2_Z6lyFStQ2-rN50_r&gclid=CjwKCAiA3fnJBhAgEiwAyqmY5bjgDamTbZn40LFp6l88wXU9t3K6ThQF5M75IHoRU4Jq0slbKWrGNxoCjGkQAvD_BwE
'''
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

#def count_typhoons_by_region(ibtracs_path, marine_shp_path, output_path):
def count_typhoons_by_region(marine_shp_path,):
    # 1. Read and pre-process Marine regions
    print("正在读取海区 Shapefile 数据...")
    gdf = gpd.read_file(marine_shp_path)

    region_col_name = 'NAME'

    target_regions = [
        'Yellow Sea', 
        'Eastern China Sea', 
        'South China Sea'
    ]

    target_keywords = ['South China Sea', 'Eastern China Sea', 'East China Sea', 
                   'Yellow Sea', 'Bo Hai', 'Taiwan Strait', 'Japan Sea']

    pattern = '|'.join(target_keywords)
    gdf_targets = gdf[gdf['NAME'].str.contains(pattern, case=False, na=False)] # Regular expressions are enabled here to implement fuzzy matching.

    if gdf_targets.crs != "EPSG:4326":
        gdf_targets = gdf_targets.to_crs("EPSG:4326")

    print(f"已筛选出海区: {gdf_targets[region_col_name].unique()}")

    # 2. Read and pre-process IBTrACS (台风数据)
    print("正在读取 IBTrACS 数据 (可能需要几秒钟)...")
    # low_memory=False 防止大文件读取时的类型警告
    df_ibtracs = pd.read_csv(ibtracs_path, low_memory=False)

    use_cols   = ['SID', 'ISO_TIME', 'LAT', 'LON']
    # remove the missing line for lat and lon
    df_clean = df_ibtracs[use_cols].dropna(subset=['LAT', 'LON'])

    # convert the time format
    df_clean['ISO_TIME'] = pd.to_datetime(df_clean['ISO_TIME'])
    df_clean['year'] = df_clean['ISO_TIME'].dt.year

    # convert to GeoDataFrame
    print("Now it is transforming the lat/lon")
    geometry = [Point(xy) for xy in zip(df_clean['LON'], df_clean['LAT'])]
    gdf_points = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")



def main():
    shapefile_path = "/mnt/f/data/World_Seas_IHO_v3/World_Seas_IHO_v3.shp" # 建议写绝对路径，防止找不到文件

    count_typhoons_by_region(shapefile_path)

if __name__ == '__main__':
    main()