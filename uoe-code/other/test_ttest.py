from scipy.stats import t
import numpy as np

# 假设的样本数据
data = [20, 22, 19, 24, 25, 18, 23, 21]

# 计算样本均值和标准差
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)

# 样本量和自由度
n = len(data)
df = n - 1

# 95%置信水平的双边检验的alpha值
alpha = 0.025  # 因为是双边，所以每边2.5%

# 计算t临界值
t_critical = t.ppf(1 - alpha, df)

# 计算标准误差
se = sample_std / np.sqrt(n)

# 计算置信区间
ci_lower = sample_mean - t_critical * se
ci_upper = sample_mean + t_critical * se

print("置信区间:", ci_lower, ci_upper)
