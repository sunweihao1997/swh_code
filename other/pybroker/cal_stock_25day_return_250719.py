import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os

sys.path.append("/home/sun/swh_code/module/")
from module_stock import cal_index_for_stock, standardize_and_normalize, map_df

file_path = "/home/sun/wd_14/data/data/other/stock_price_single/"
file_all = os.listdir(file_path)
for i in file_all:
    if not i.endswith(".xlsx"):
        continue
    print(i)
    df = pd.read_excel(f"{file_path}{i}")

    # (1) 未来5-15天平均收盘价
    future_prices = df['收盘'].shift(-15).rolling(window=11).mean()

    # (2) 过去3天平均收盘价 (t-2, t-1, t)
    past_prices = df['收盘'].rolling(window=3).mean()

    # 第一次平滑
    df['close_smooth_1'] = df['收盘'].rolling(window=3).mean()

    # 第二次平滑（对第一次平滑结果再做一次平滑）
    df['close_smooth_2'] = df['close_smooth_1'].rolling(window=5).mean()

    # (3) 计算收益率
    df['future_5_15'] = future_prices
    df['past_3_return'] = past_prices
    df['future_5_15_vs_past_3_return'] = future_prices / past_prices - 1
    df['return_25'] = df['close_smooth_2'].shift(-25) / df['close_smooth_2'] - 1


    
    col_map = {
        "股票代码": "code",
        "名称": "name",
        "日期": "date",
        "市盈率": "pe_ratio",
        "市盈率-动态": "pe_ratio_dynamic",
        "市盈率(动)": "pe_ratio_dynamic",
        "市盈率_TTM": "pe_ratio_ttm",
        "收盘": "close",
        "收盘_z": "close_z",
        "成交量": "volume",
        "成交量_z": "volume_z",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "涨跌幅": "pct_change",
        # 你可以继续添加其他需要的列名映射
    }

    # 应用映射替换列名
    df.rename(columns=col_map, inplace=True)
    df.to_excel(f"/home/sun/wd_14/data/data/other/stock_25_return/{i}", index=False)

