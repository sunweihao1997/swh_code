'''
2025-12-19 (Updated)
This script calculates the count how many typhoon locates in corresponding marine zone,
calculates climatology (annual average), and plots the results.
'''
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

def count_typhoons_by_region(ibtracs_path, marine_shp_path, output_csv_path, output_fig_path):
    # ---------------------------------------------------------
    # 1. Read and pre-process Marine regions
    # ---------------------------------------------------------
    print("正在读取海区 Shapefile 数据...")
    gdf = gpd.read_file(marine_shp_path)

    region_col_name = 'NAME'

    # 定义目标海区关键词
    target_keywords = ['South China Sea', 'Eastern China Sea', 'East China Sea', 
                   'Yellow Sea', 'Bo Hai', 'Taiwan Strait', 'Japan Sea']

    pattern = '|'.join(target_keywords)
    # 筛选海区 (使用正则模糊匹配)

    print(gdf['NAME'])
    gdf_targets = gdf[gdf['NAME'].str.contains(pattern, case=False, na=False)].copy()

    # 统一坐标系为 WGS84
    if gdf_targets.crs != "EPSG:4326":
        gdf_targets = gdf_targets.to_crs("EPSG:4326")

    print(f"已筛选出海区: {gdf_targets[region_col_name].unique()}")

    # ---------------------------------------------------------
    # 2. Read and pre-process IBTrACS (台风数据)
    # ---------------------------------------------------------
    print("正在读取 IBTrACS 数据 (可能需要几秒钟)...")
    df_ibtracs = pd.read_csv(ibtracs_path, low_memory=False, skiprows=[1]) 

    use_cols = ['SID', 'ISO_TIME', 'LAT', 'LON']
    df_clean = df_ibtracs[use_cols].dropna(subset=['LAT', 'LON'])

    # 时间格式转换
    df_clean['ISO_TIME'] = pd.to_datetime(df_clean['ISO_TIME'], errors='coerce')
    df_clean['year'] = df_clean['ISO_TIME'].dt.year

    # 转换为 GeoDataFrame
    print("正在转换经纬度坐标...")
    geometry = [Point(xy) for xy in zip(df_clean['LON'], df_clean['LAT'])]
    gdf_points = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4326")

    # ---------------------------------------------------------
    # 3. Count typhoons in each region (Spatial Join)
    # ---------------------------------------------------------
    print("正在统计海区内的台风数量 (Spatial Join)...")
    # 空间连接：判断台风点是否在海区多边形内
    joined = gpd.sjoin(gdf_points, gdf_targets[[region_col_name, 'geometry']], how="inner", predicate="within")

    # 统计每年、每个海区的唯一台风编号 (SID) 数量
    result = joined.groupby(['year', region_col_name])['SID'].nunique().reset_index()
    result.columns = ['Year', 'Region', 'Typhoon_Count']
    result = result.sort_values(by=['Year', 'Region'])

    # 保存年度统计结果
    result.to_csv(output_csv_path, index=False)
    
    # 转为透视表 (行:年份, 列:海区, 值:数量)，缺失年份填0
    pivot_table = result.pivot(index='Year', columns='Region', values='Typhoon_Count').fillna(0)
    
    # ---------------------------------------------------------
    # 4. Calculate Climatology (计算气候态)
    # ---------------------------------------------------------
    print("正在计算气候态...")
    # 计算每列（每个海区）的均值
    climatology = pivot_table.mean()
    # 将 Series 转为 DataFrame 以便合并，列名设为 'Clim_Val'
    clim_df = climatology.reset_index() 
    clim_df.columns = [region_col_name, 'Clim_Val']

    print("气候态结果 (年平均个数):")
    print(clim_df)

    # ---------------------------------------------------------
    # 5. Visualization (绘图)
    # ---------------------------------------------------------
    print("正在绘图...")
    
    # 将气候态数据合并回地理数据中，以便绘图
    # 注意：gdf_targets 可能包含同名的多个多边形，merge 会自动匹配
    gdf_plot = gdf_targets.merge(clim_df, on=region_col_name, how='left')

    # 创建画布
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制背景海区 (根据气候态数值着色，可选)
    gdf_plot.plot(ax=ax, column='Clim_Val', cmap='Blues', edgecolor='black', 
                  legend=True, legend_kwds={'label': "Annual Avg Typhoons"})

    # 在图上标注：海区名 + 数量
    # 为了避免同一个海区如果有多个碎片多边形导致重复标注，我们可以先按名字 dissolve 一下计算中心点，或者简单遍历
    # 这里采用简单遍历，但为了防止太密集的点，可以考虑只取最大的 geometry (此处为简化直接遍历)
    
    processed_labels = set() # 用于去重标签，防止多岛屿造成标签重叠

    for idx, row in gdf_plot.iterrows():
        region_name = row[region_col_name]
        val = row['Clim_Val']
        
        # 如果该海区没有数据（NaN），则跳过或显示0
        if pd.isna(val):
            continue

        # 获取几何中心用于放置文字
        centroid = row.geometry.centroid
        
        # 简单的去重逻辑：如果该名字已经标过，且距离很近，就不标了？
        # 或者更简单：直接标，但字体设小一点。
        # 此处逻辑：构造显示文本
        label_text = f"{region_name}\n({val:.1f})"
        
        # 标注文字
        ax.annotate(text=label_text, 
                    xy=(centroid.x, centroid.y), 
                    xytext=(0, 0), 
                    textcoords="offset points", 
                    ha='center', va='center',
                    fontsize=9, fontweight='bold', color='darkred',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))

    ax.set_title(f"Typhoon Climatology by Marine Region ({pivot_table.index.min()}-{pivot_table.index.max()})", fontsize=15)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    # 保存图片
    plt.savefig(output_fig_path, dpi=300, bbox_inches='tight')
    
    print("-" * 30)
    print(f"处理完成！")
    print(f"CSV结果已保存至: {output_csv_path}")
    print(f"气候态图片已保存至: {output_fig_path}")
    print("-" * 30)

def main():
    # 请根据实际环境修改路径
    shapefile_path = "/home/sun/data/download_data/World_Seas_IHO_v3/World_Seas_IHO_v3.shp" 
    output_csv_path = "/home/sun/data/process/analysis/typhoon_prediction/typhoon_count.csv"
    # 新增图片输出路径
    output_fig_path = "/home/sun/data/process/analysis/typhoon_prediction/typhoon_climatology.png"
    
    ibtracs_csv_path = "/home/sun/data/download_data/IBTRACS_typhoon/ibtracs.WP.list.v04r01.csv" 

    count_typhoons_by_region(ibtracs_csv_path, shapefile_path, output_csv_path, output_fig_path)

if __name__ == '__main__':
    main()