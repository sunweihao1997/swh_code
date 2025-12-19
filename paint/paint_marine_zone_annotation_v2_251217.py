'''
2025-12-17
This script is to plot the boudaries of the marine zone
'''
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe  # <--- 1. 显式导入这个模块

# 1. 读取数据
shapefile_path = "/mnt/f/data/World_Seas_IHO_v3/World_Seas_IHO_v3.shp" # 建议写绝对路径，防止找不到文件
# 如果你的shp文件不在这个路径，请改回原来的写法: "World_Seas_IHO_v3.shp"
gdf = gpd.read_file(shapefile_path)

# --- 筛选海域 ---
# 根据你刚才的运行结果，Bo Hai 和 Taiwan Strait 没找到，可能是被包含在其他海域里了
# 我们先画出找到的这4个
target_keywords = ['South China Sea', 'Eastern China Sea', 'East China Sea', 
                   'Yellow Sea', 'Bo Hai', 'Taiwan Strait', 'Japan Sea']

pattern = '|'.join(target_keywords)
target_seas = gdf[gdf['NAME'].str.contains(pattern, case=False, na=False)]

print("成功筛选出以下海域：")
print(target_seas['NAME'].unique())

# 2. 开始绘图
fig, ax = plt.subplots(figsize=(15, 12))

# 绘制底图（浅灰色）
gdf.plot(ax=ax, color='#f0f0f0', edgecolor='white')

# 绘制目标海域（使用 Set2 配色）
target_seas.plot(ax=ax, column='NAME', legend=True, cmap='Set2', alpha=0.9, edgecolor='black')

# 3. 添加名字标注
for idx, row in target_seas.iterrows():
    centroid = row.geometry.centroid
    text = row['NAME']
    
    # 简单的换行处理
    if text == "Eastern China Sea":
        display_text = "East China\nSea"
    elif text == "South China Sea":
        display_text = "South China\nSea"
    else:
        display_text = text
        
    plt.annotate(text=display_text, 
                 xy=(centroid.x, centroid.y),
                 horizontalalignment='center', 
                 verticalalignment='center',
                 fontsize=10, 
                 color='black', 
                 weight='bold',
                 # <--- 2. 这里使用了显式导入的 pe，修复了 AttributeError
                 path_effects=[pe.withStroke(linewidth=3, foreground="white")])

# 4. 设置范围并保存
ax.set_xlim(100, 150)
ax.set_ylim(0, 50)

plt.title("Seas Bordering China", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# 保存图片到当前目录
save_path = "/mnt/f/wsl_plot/donghai/typhoon_prediction/China_Bordering_Seas.png"
plt.savefig(save_path, dpi=300)
print(f"绘图完成！图片已保存为: {save_path}")

# plt.show() # <--- 3. 在Linux无界面环境下注释掉这一行，避免 qt.qpa 报错