'''
2025-7-3
This script is a test for the stock monitoring script, which is used to monitor the stock data
'''
import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas_ta as ta

# Get all the stock information
spot_df = pd.read_excel("/home/sun/data/other/stock_realtime_data.xlsx", dtype={"代码": str})
#print(spot_df[spot_df['代码'] == '300957'])

# Get the historical data for a specific stock
end_date = datetime.today().strftime("%Y%m%d")
start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

#df = ak.stock_zh_a_hist(symbol='300957', start_date=start_date, end_date=end_date,)
#df.to_csv("/home/sun/data/other/stock_price_single/300957.csv", index=False)

#df = pd.read_csv("/home/sun/data/other/stock_price_single/300957.csv")
#print(df)
df = ak.stock_zh_a_hist(symbol="300957", adjust="qfq")

df['CCI'] = ta.cci(high=df['最高'], low=df['最低'], close=df['收盘'], length=14)
print(df['CCI']) # Correct

macd = ta.macd(df['收盘'])

df = pd.concat([df, macd], axis=1)

# 查看结果
#print(df[['日期', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9']].tail(20)) # 虽然数值跟东方财富对不上，但是趋势是对的

# 计算 OBV
df['OBV'] = ta.obv(close=df['收盘'], volume=df['成交量'])

# 查看结果
print(df[['日期', '收盘', '成交量', 'OBV']].tail())