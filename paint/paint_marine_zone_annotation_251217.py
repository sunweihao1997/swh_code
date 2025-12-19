'''
2025-12-17
This script is to plot the boudaries of the marine zone
'''
import geopandas as gpd
import matplotlib.pyplot as plt

# 1. 读取数据
# 注意：只需要读取 .shp 文件，它会自动寻找同名的 .dbf 等文件
shapefile_path = "/mnt/f/data/World_Seas_IHO_v3/World_Seas_IHO_v3.shp"
gdf = gpd.read_file(shapefile_path)

# 2. 筛选出“中国南海”和“中国东海”
# IHO数据中的名称通常是英文，"South China Sea" 和 "East China Sea"
target_seas = gdf[gdf['NAME'].isin(['South China Sea', 'East China Sea'])]

# 3. 开始绘图
fig, ax = plt.subplots(figsize=(15, 10))

# 绘制世界所有海域的底图（浅灰色，作为背景）
gdf.plot(ax=ax, color='lightgrey', edgecolor='white', alpha=0.5)

# 突出显示目标海域（红色或蓝色）
target_seas.plot(ax=ax, column='NAME', legend=True, cmap='Set1', alpha=0.8, edgecolor='black')

# 4. 添加标注
for x, y, label in zip(target_seas.geometry.centroid.x, target_seas.geometry.centroid.y, target_seas.geometry):
    # 这里需要根据你的具体行找到对应的名字，简单起见演示标注
    # 实际操作中可以使用 target_seas.apply 自动标注
    pass 
    
# 这里做一个简单的循环标注名字
for idx, row in target_seas.iterrows():
    plt.annotate(text=row['NAME'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
                 horizontalalignment='center', fontsize=12, color='black', weight='bold')

# 设置标题
plt.title("Visualization of East & South China Sea (IHO Data)", fontsize=15)

# 限制显示范围（聚焦在亚洲区域，避免图太大）
# 你可以根据需要调整这个经纬度范围
ax.set_xlim(90, 140)
ax.set_ylim(0, 45)

plt.savefig("/mnt/f/wsl_plot/donghai/typhoon_prediction/marin_zone_annotation.png")