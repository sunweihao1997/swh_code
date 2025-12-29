'''
2025-12-17
This script is to plot the boudaries of the marine zone

v4: same with v3, but run on N100 Huaibei
'''
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe  # <--- 1. 显式导入这个模块

# 1. 读取数据
shapefile_path = "/home/sun/data/download_data/World_Seas_IHO_v3/World_Seas_IHO_v3.shp" # 建议写绝对路径，防止找不到文件
# 如果你的shp文件不在这个路径，请改回原来的写法: "World_Seas_IHO_v3.shp"
gdf = gpd.read_file(shapefile_path)

# --- 关键改动：定义地理范围并进行空间切片 ---
# 我们不再根据名字筛选，而是根据地理位置筛选
# 定义地图的显示范围 (西北太平洋区域)
xmin, xmax = 100, 150
ymin, ymax = 0, 50

# 使用 .cx 索引器快速筛选出与这个边界框相交的所有海域
# 这比循环检查每个海域的中心点要快得多且准确
visible_seas = gdf.cx[xmin:xmax, ymin:ymax].copy()

print(f"在此范围内共找到 {len(visible_seas)} 个海域图斑。")
# print(visible_seas['NAME'].unique()) # 如果想看具体有哪些名字可以取消注释

# 2. 开始绘图
fig, ax = plt.subplots(figsize=(16, 12)) #稍微调大一点画布

# 绘制全球底图（作为背景，浅灰色，防止边缘有空白）
gdf.plot(ax=ax, color='#e0e0e0', edgecolor='white', linewidth=0.5)

# 绘制筛选出来的可见海域
# column='NAME': 根据名字分配颜色
# cmap='tab20': 使用一个颜色更丰富的色板，因为现在海域变多了
# legend=False: 关闭图例，因为海域太多，图例会把图挡住，我们直接用地图上的文字标注
visible_seas.plot(ax=ax, column='NAME', cmap='tab20', alpha=0.8, edgecolor='black', linewidth=0.8, legend=False)

# 3. 添加名字标注
# 只遍历筛选出来的海域进行标注
for idx, row in visible_seas.iterrows():
    # 计算中心点
    # 对于一些形状极不规则的海域，centroid可能在外面，改用 representative_point() 确保点在多边形内部
    centroid = row.geometry.representative_point()
    text = row['NAME']
    
    # 简单的换行处理，让长名字显示更好看
    if text == "Eastern China Sea":
        display_text = "East China\nSea"
    elif text == "South China Sea":
        display_text = "South China\nSea"
    elif text == "North Pacific Ocean":
        display_text = "North Pacific\nOcean"
    elif text == "Philippine Sea":
        display_text = "Philippine\nSea"
    elif len(text) > 15 and ' ' in text: # 对于其他过长的名字，尝试在空格处换行
        display_text = text.replace(' ', '\n', 1)
    else:
        display_text = text
        
    # 只有当中心点在我们的视野范围内才进行标注，避免边缘的文字标到图外面
    if xmin < centroid.x < xmax and ymin < centroid.y < ymax:
        plt.annotate(text=display_text, 
                     xy=(centroid.x, centroid.y),
                     horizontalalignment='center', 
                     verticalalignment='center',
                     fontsize=9, # 字体稍微调小一点以适应更多标签
                     color='black', 
                     weight='bold',
                     # 加白色描边，确保在深色背景上也能看清
                     path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# 4. 严格设置显示范围（保持不动）
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

plt.title("Marine Areas in the Northwest Pacific (IHO v3)", fontsize=16)
plt.xlabel("Longitude")
plt.ylabel("Latitude")

# 保存图片
save_path = "Northwest_Pacific_Seas_All_Colored.png"
# 保存图片到当前目录
save_path = "/home/sun/paint/donghai/typhoon_prediction/Northwest_Pacific_Seas_All_Colored.png"
plt.savefig(save_path, dpi=300)
print(f"绘图完成！图片已保存为: {save_path}")

# plt.show() # <--- 3. 在Linux无界面环境下注释掉这一行，避免 qt.qpa 报错