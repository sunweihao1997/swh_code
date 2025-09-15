'''
20250902
This script is to test the ATR shrinking pattern detection for stocks:cal_stock_ATR_shrinking_250901
'''
from cal_stock_ATR_shrinking_250901 import low_volatility_flags, link_week_daily  # 如果函数在这个文件里
import akshare as ak
import pandas as pd
import numpy as np

# 1) 拉数据（建议用前复权）
stock = "000001"
df = ak.stock_zh_a_hist(
    symbol=stock, period="daily",
    start_date="20200902", end_date="20250902",
    adjust="qfq"  # 前复权，更适合技术分析；如需原始价可用 ""
)

# 2) 重命名为标准 OHLCV
df = df.rename(columns={
    '日期': 'Date',
    '开盘': 'Open',
    '最高': 'High',
    '最低': 'Low',
    '收盘': 'Close',
    '成交量': 'Volume',
    # 其他列（如 成交额、涨跌幅）可保留，不影响后续
})

# 3) 设索引 + 排序
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()

# 4) 转为数值（有些列可能是字符串）
for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 5) 可选：把"手"转为股（后续若要算成交额更准确）
# df['Volume'] = df['Volume'] * 100

# 6) 计算日/周低波标签（pandas_ta 版 ATR / BBWidth）
#    若你的 low_volatility_flags 用到了 pandas_ta 指标，请确保 pandas_ta 已安装
flags = low_volatility_flags(
    df,
    d_len=20, w_len=20,
    d_lookback_days=252,   # ≈12个月
    w_lookback_weeks=52,   # ≈12个月
    d_pct=0.20, w_pct=0.35,
    bb_std=2.0
)

# 7) 周线背景 + 日线执行 联动，生成候选日（布尔序列）
candidates = link_week_daily(
    df, flags['daily'], flags['weekly'],
    exec_window_days=20  # 周线触发后给20个交易日的执行窗口
)

# 8) 查看最近结果
out = pd.DataFrame({
    'Close': df['Close'],
    'daily_low_vol': flags['daily']['daily_low_vol'].reindex(df.index),
    'weekly_low_vol_on_weekly_index': flags['weekly']['weekly_low_vol']  # 周五索引上的标签
})
out['candidate'] = candidates
print(out.tail(15))

# 9) 需要保存的话
out.to_csv(f'./{stock}_lowvol_candidates.csv', encoding='utf-8-sig')
