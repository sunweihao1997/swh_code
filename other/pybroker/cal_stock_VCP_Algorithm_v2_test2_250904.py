from cal_stock_VCP_Algorithm_v2_250904 import find_swings_zigzag, ensure_datetime_index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# -----------------------------
# 1) 准备数据
# -----------------------------
csv_path = "/home/ubuntu/stock_data/stock_price_test/301528.csv"
symbol = os.path.splitext(os.path.basename(csv_path))[0]  # 从文件名提取代码

df = pd.read_csv(csv_path)


# 按你之前的重命名习惯（注意大小写要对得上 CSV）
df = df.rename(columns={
    "date": "Date",
    "high": "High",
    "low": "Low",
    "close": "Close"
})[["Date", "High", "Low", "Close"]]

# 转换时间索引
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"]).set_index("Date")
df = df.astype(float).sort_index()

# -----------------------------
# 2) 调用 ZigZag 函数
# -----------------------------
swings = find_swings_zigzag(
    df,
    threshold=0.1,       # 6% 阈值
    mode="percent",
    use_high_low=True,
    include_last=True     # 最后一个候选点也画出来
)

# 合并峰谷，得到连续折线
pivots = swings["swing_high"].combine_first(swings["swing_low"]).dropna()

# -----------------------------
# 3) 绘图
# -----------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Close", linewidth=1.0)

# Swing 高点
hi = swings["swing_high"].dropna()
plt.scatter(hi.index, hi.values, marker="^", color="red", s=60, label="Swing High")

# Swing 低点
lo = swings["swing_low"].dropna()
plt.scatter(lo.index, lo.values, marker="v", color="green", s=60, label="Swing Low")

# ZigZag 连线
plt.plot(pivots.index, pivots.values, linewidth=1.4, color="orange", label="ZigZag")

plt.title(f"{symbol} ZigZag swings (threshold=6%)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

out_dir = "/home/ubuntu/plot"
os.makedirs(out_dir, exist_ok=True)
plt.savefig(os.path.join(out_dir, f"{symbol}_zigzag_swings.png"), dpi=300)