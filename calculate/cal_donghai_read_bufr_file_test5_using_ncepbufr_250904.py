import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# 读取插值后的 NetCDF
ds = xr.open_dataset("/mnt/f/amv_grid.nc")

# 提取经纬度和风场
lon = ds["lon"].values
lat = ds["lat"].values
u = ds["u"].values
v = ds["v"].values

# 建立网格
lon2d, lat2d = np.meshgrid(lon, lat)

plt.figure(figsize=(12, 6))

# 画底图（可选，填充风速强度）
speed = (u**2 + v**2) ** 0.5
plt.contourf(lon2d, lat2d, speed, levels=20, cmap="jet")

# 画矢量箭头（每隔 n 个点画一个，避免太密集）
step = 5
plt.quiver(lon2d[::step, ::step], lat2d[::step, ::step],
           u[::step, ::step], v[::step, ::step],
           scale=500)

# 坐标轴和标题
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("AMV Wind Vectors (gridded)")

plt.colorbar(label="Wind speed (m/s)")
plt.tight_layout()
plt.show()
