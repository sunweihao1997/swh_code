'''
2025-9-7
This script is used to test the VCP strategy implementation in pybroker.

Three sub-script:
`cal_stock_ATR_shrinking_250901.py`
`cal_stock_VCP_Algorithm_v2_250904.py`
`cal_stock_volume_energy_break_v2_250904.py`
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import akshare as ak
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from cal_stock_ATR_shrinking_250901 import atr_ta, bb_width_ta, ensure_datetime_index, link_week_daily, to_weekly, percentile_rank_last, low_volatility_flags
from cal_stock_VCP_Algorithm_v2_250904 import find_swings_zigzag
from cal_stock_VCP_Algorithm_v2_250904 import extract_contractions_v2, is_valid_vcp_sequence_v2

# ================= The test stock ===================
# Here I use 002245 for testing
stock_code = '002245'
stock_data = pd.read_excel(f'/home/sun/data_n100/stocks_price/{stock_code}.xlsx', parse_dates=True) #parse_date: 告诉 pandas 尝试把可以解析的字符串列转换为 日期时间类型 (datetime64)
#stock_data = pd.read_excel(f'/home/sun/data_n100/stocks_price/{stock_code}.xlsx',)
#print(stock_data.head())

# ================= 1. Test Part for ATR Shrinking ===================
# ================== 1.1 Calculate Indicators ===================
flags = low_volatility_flags(df_daily=stock_data)

daily_flags = flags['daily']
weekly_flags = flags['weekly']

candidates = link_week_daily(
    daily_df=stock_data,
    flags_d=daily_flags,
    flags_w=weekly_flags,
    exec_window_days=500,   # ATR收缩后观察的窗口
    start_next_day=True    # 窗口从下一日开始
)



# ================= 1.2 Plot ===================
def plot_candidates(df, candidates, title="Stock Price with Candidate Days"):
    """
    绘制日线股价，并在候选交易日上标记。
    
    参数：
    - df: 日线数据，必须有 ['date','open','high','low','close']
    - candidates: 布尔 Series，与 df.index 对齐
    - title: 图标题
    """

    df = df.copy()
    #print(df.columns)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    # 收盘价
    close = df['close']
    
    fig, ax = plt.subplots(figsize=(16,6))
    
    # 1️⃣ 绘制收盘价曲线
    ax.plot(close.index, close.values, label='Close Price', color='blue', linewidth=1.5)
    
    # 2️⃣ 标记候选交易日
    candidate_dates = candidates[candidates].index
    candidate_prices = close.loc[candidate_dates]
    ax.scatter(candidate_dates, candidate_prices, color='red', s=50, label='Candidate Days', zorder=5, marker='o')

    # 3️⃣ 美化 x 轴日期
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    # 4️⃣ 标题和图例
    ax.set_title(title, fontsize=16)
    ax.set_ylabel("Price")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'/home/sun/paint/stock/{stock_code}_candidates.png')

# =================2. Test Part for VCP Strategy ===================
# ================== 2.1 Calculate ZIGZIG LOW/High SWINGS ===================

# 2.1.1 Merge the Stock data and Candidate days data
candidates_df = candidates.rename('candidate').reset_index()
if candidates_df.columns[0] != 'date':
    print("Renaming the first column to 'date'")
    candidates_df = candidates_df.rename(columns={candidates_df.columns[0]: 'date'})
# 合并到 stock_data
stock_data = stock_data.merge(candidates_df, on='date', how='left')

#print(stock_data.head())

# 2.2.2 Calculate ZIGZAG LOW/High SWINGS
zigzag_result = find_swings_zigzag(
    df=stock_data,
    threshold=0.05,  # 7% 的阈值
    )
#with pd.option_context('display.max_rows', 500, 'display.max_columns', None):
#    print(zigzag_result.head(500))

# 2.2.3 Extract Contractions and Validate VCP Patterns
res_pairs = extract_contractions_v2(
    df=stock_data,
    swings=zigzag_result,
    lookback=None,               # 用全历史（也可改 120/180）
    min_drop=0.01,               # 最小回撤3%
    max_drop=0.40,               # 最大回撤40%
    min_bars_per_leg=1,          # 每段至少5根K，抑制锯齿
    max_last_drop=None           # 如需“末段更紧”限制，可给 0.10 等
)
print(f"[VCP] pairs={len(res_pairs['pairs'])}, last_drop={res_pairs['last_drop']}")


if __name__ == "__main__":
    plot_candidates(stock_data, candidates, title=f"Stock {stock_code} Price with Candidate Days")