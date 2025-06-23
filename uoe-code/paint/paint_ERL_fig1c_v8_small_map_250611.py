import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle

# 创建投影，使用 PlateCarree (经纬度投影)
projection = ccrs.PlateCarree()

# 创建绘图
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=projection)

# 设置地图范围：印度地区 (大致经纬度范围)
ax.set_extent([67, 97, 6, 36], crs=projection)

# 添加地理特征
# 海岸线
ax.add_feature(cfeature.COASTLINE, linewidth=2, edgecolor='gray')
# 国家边界
ax.add_feature(cfeature.BORDERS, linewidth=2, edgecolor='gray')
# 州或省边界
#ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='black')

# 添加网格线
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.7, linestyle='--')
gl.xlabel_style = {'size': 20, 'color': 'gray'}
gl.ylabel_style = {'size': 20, 'color': 'gray'}

# 在地图上添加一个矩形框
# 矩形框范围：76° to 87°E, 20° to 28°N
rect = Rectangle((75, 16), 87 - 75, 27 - 16, 
                 linewidth=3.5, edgecolor='orange', facecolor='none', transform=ccrs.PlateCarree())
ax.add_patch(rect)

# 添加标题
#plt.title("Map of India with a Bounding Box (76° to 87°E, 20° to 28°N)", fontsize=14)

plt.savefig('/home/sun/paint/ERL/Article_fig1c_add_small_map_v2_20260611.pdf')