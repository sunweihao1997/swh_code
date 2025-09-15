'''
2025-8-30
This script tries to decompose stock price data using EEMD method.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PyEMD import EEMD
import akshare as ak

df = ak.stock_zh_a_hist(symbol="600900", start_date="20200101", end_date="20250830", adjust="qfq")
stock_prices = df['收盘']  # 假设'Close'是收盘价列

# 2. 实例化EEMD对象
eemd = EEMD()

# 3. 对股价数据进行EEMD分解
IMFs = eemd.eemd(stock_prices.values)
#print(len(IMFs), IMFs.shape)
# 4. 绘制每个IMF的时间序列
plt.figure(figsize=(12, 20))

# 绘制原始股价
plt.subplot(len(IMFs) + 2, 1, 1)  # 原始股价
plt.plot(df['日期'], stock_prices, label="Original Stock Prices", color='blue')
plt.title("Original Stock Prices")
plt.xticks(rotation=45)
plt.legend()

# 绘制每个IMF的时间序列
for i, imf in enumerate(IMFs):
    plt.subplot(len(IMFs) + 2, 1, i + 2)  # 每个IMF的时间序列
    plt.plot(df['日期'], imf, label=f"IMF {i + 1}", color='orange')
    plt.title(f"IMF {i + 1}")
    plt.xticks(rotation=45)
    plt.legend()

# 绘制残余部分（最后的趋势）
residual = stock_prices.values - np.sum(IMFs, axis=0)
plt.subplot(len(IMFs) + 2, 1, len(IMFs) + 2)  # 残余部分
plt.plot(df['日期'], residual, label="Residual", color='green')
plt.title("Residual")
plt.xticks(rotation=45)
plt.legend()

# 调整布局，防止重叠
plt.tight_layout()
plt.savefig("/home/ubuntu/plot/EEMD_decompose_600582_250830.png")