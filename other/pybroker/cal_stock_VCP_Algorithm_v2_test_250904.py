# pip install -U akshare pandas numpy matplotlib
import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# 1) 工具函数
# ------------------------------
def ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    _df = df.copy()
    if not isinstance(_df.index, pd.DatetimeIndex):
        if "Date" in _df.columns:
            _df["Date"] = pd.to_datetime(_df["Date"], errors="coerce")
            _df = _df.dropna(subset=["Date"]).set_index("Date")
        else:
            raise ValueError("DataFrame 需要 DatetimeIndex 或包含 'Date' 列")
    _df = _df[~_df.index.duplicated(keep="last")].sort_index()
    return _df

def find_swings_zigzag(
    df: pd.DataFrame,
    threshold: float = 0.05,
    *,
    mode: str = "percent",     # "percent" 或 "abs"
    use_high_low: bool = True,
    include_last: bool = False
) -> pd.DataFrame:
    if "Close" not in df.columns:
        raise ValueError("DataFrame 必须包含 'Close' 列。")
    price_h = df["High"] if (use_high_low and "High" in df.columns) else df["Close"]
    price_l = df["Low"]  if (use_high_low and "Low"  in df.columns) else df["Close"]
    close = df["Close"]

    n = len(df)
    if n == 0:
        return pd.DataFrame({"swing_high": [], "swing_low": []}, index=df.index)

    idx = df.index
    swing_high = np.full(n, np.nan, dtype=float)
    swing_low  = np.full(n, np.nan, dtype=float)

    direction = 0  # 0=未定, 1=上涨段（找峰）, -1=下跌段（找谷）
    max_price = price_h.iloc[0]; max_i = 0
    min_price = price_l.iloc[0]; min_i = 0

    if mode == "percent":
        def risen_from_min(curr_high, base_min): return (curr_high - base_min) / base_min >= threshold
        def pulled_from_max(curr_low, base_max): return (base_max - curr_low) / base_max >= threshold
    elif mode == "abs":
        def risen_from_min(curr_high, base_min): return (curr_high - base_min) >= threshold
        def pulled_from_max(curr_low, base_max): return (base_max - curr_low) >= threshold
    else:
        raise ValueError("mode 只能为 'percent' 或 'abs'。")

    for i in range(1, n):
        hi = price_h.iloc[i]; lo = price_l.iloc[i]

        if direction == 0:
            if hi > max_price: max_price, max_i = hi, i
            if lo < min_price: min_price, min_i = lo, i
            if risen_from_min(hi, min_price) and min_i < i:
                swing_low[min_i] = min_price; direction = 1
                max_price, max_i = hi, i
            elif pulled_from_max(lo, max_price) and max_i < i:
                swing_high[max_i] = max_price; direction = -1
                min_price, min_i = lo, i
            continue

        if direction == 1:
            if hi > max_price: max_price, max_i = hi, i
            if pulled_from_max(lo, max_price) and max_i < i:
                swing_high[max_i] = max_price; direction = -1
                min_price, min_i = lo, i
        else:  # direction == -1
            if lo < min_price: min_price, min_i = lo, i
            if risen_from_min(hi, min_price) and min_i < i:
                swing_low[min_i] = min_price; direction = 1
                max_price, max_i = hi, i

    if include_last:
        if direction == 1:
            swing_high[max_i] = max_price
        elif direction == -1:
            swing_low[min_i] = min_price

    return pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low}, index=idx)

# ------------------------------
# 2) 用 akshare 获取数据（示例：贵州茅台 600519，前复权）
# ------------------------------
symbol = "600519"     # A股代码，不带交易所前缀
start  = "20220101"
end    = "20251231"

df_raw = ak.stock_zh_a_hist(
    symbol=symbol, period="daily",
    start_date=start, end_date=end, adjust="qfq"   # qfq: 前复权；不复权设 adjust=None
)

print(df_raw.head())

## akshare 返回中文列名，这里统一成英文字段
## 常见列：日期/开盘/收盘/最高/最低/成交量/成交额/振幅/涨跌幅/涨跌额/换手率
#df = df_raw.rename(columns={
#    "日期": "Date", "最高": "High", "最低": "Low", "收盘": "Close"
#})[["Date", "High", "Low", "Close"]]
#df = ensure_datetime_index(df).astype(float)
#
## ------------------------------
## 3) ZigZag 标注
## ------------------------------
#swings = find_swings_zigzag(
#    df, threshold=0.06,         # 例如 6% 阈值；可按波动调整
#    mode="percent",
#    use_high_low=True,
#    include_last=False
#)
#
## 合并便于画线：把峰/谷合成一个 pivots 序列并去掉 NaN
#pivots = swings["swing_high"].combine_first(swings["swing_low"]).dropna()
#
## ------------------------------
## 4) 可视化
## ------------------------------
#plt.figure(figsize=(12, 6))
#plt.plot(df.index, df["Close"], label="Close", linewidth=1.2)
#
## Swing 点：高点▲、低点▼
#hi = swings["swing_high"].dropna()
#lo = swings["swing_low"].dropna()
#plt.scatter(hi.index, hi.values, marker="^", s=60, label="Swing High", zorder=3)
#plt.scatter(lo.index, lo.values, marker="v", s=60, label="Swing Low", zorder=3)
#
## ZigZag 连线（连接确认的拐点）
#plt.plot(pivots.index, pivots.values, linewidth=1.4, label="ZigZag")
#
#plt.title(f"{symbol} ZigZag swings (threshold=6%)")
#plt.xlabel("Date"); plt.ylabel("Price")
#plt.legend()
#plt.grid(alpha=0.3)
#plt.tight_layout()
#plt.savefig("/home/ubuntu/plots/zigzag_swings.png", dpi=300)
#