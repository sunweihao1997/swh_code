import numpy as np
from sklearn.linear_model import LinearRegression

# CCI 值（索引 474 到 483）
y = np.array([
    37.788929,
    17.159940,
    -2.942943,
    -41.223556,
    -100.236035,
    -33.704375,
    -60.078921,
    -72.488170,
    15.831594,
    109.083998
])

# x 为时间轴：0~9
x = np.arange(len(y)).reshape(-1, 1)

# 拟合
model = LinearRegression().fit(x, y)
slope = model.coef_[0]

print(f"线性回归斜率为：{slope:.6f}")
