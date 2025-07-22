import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed

def slope_upward(series: pd.Series, days: int = 10) -> bool:
    """判断最近 N 天均线是否上升（简单斜率法）"""
    if len(series) < days + 1 or series.isna().any():
        return False
    return (series.iloc[-1] - series.iloc[-(days + 1)]) / days > -0.1

import numpy as np

def slope_upward_regression(series: pd.Series, days: int = 10) -> bool:
    if len(series) < days or series[-days:].isna().any():
        return False
    y = series[-days:].values
    x = np.arange(days)
    coef = np.polyfit(x, y, 1)[0]   # 拟合一次多项式，取斜率部分
    print(coef)
    if coef > 0:
        print('Detected upward slope:', coef)
        return True
    else:
        return False
    
def calculate_slope(series: pd.Series) -> float:
    """返回给定序列的线性回归斜率"""
    y = series.dropna().values
    if len(y) < 2:
        return 0
    X = np.arange(len(y)).reshape(-1, 1)
    model = LinearRegression().fit(X, y.reshape(-1, 1))
    return model.coef_[0][0]


def check_stock_MA(code: str, name: str, pe: float, start_date: str, end_date: str):
    import os
    import pandas_ta as ta

    if pe >= 35:
        #print(f"{code} {name} 被排除: PE >= 35 ({pe:.2f})")
        return None
    else:      
        out_path = "/home/sun/data/other/stock_price_single/" # write price to this path
        file_path = f"{out_path}{code}.xlsx"

        if os.path.exists(file_path):   
            #print(f"Reading existing data for {code} {name} from {file_path}")
            df = pd.read_excel(file_path)
        else:
#            pass
            print(f"Fetching data for {code} {name} from AkShare")
            df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date,)
            df.to_excel(file_path, index=False)

    # Start calculating the Technical Indicators
    df['MA5']  = df['收盘'].rolling(window=5).mean()
    df['MA20'] = df['收盘'].rolling(window=20).mean()
    df['MA60'] = df['收盘'].rolling(window=55).mean()
    df['obv']  = ta.obv(close=df["收盘"], volume=df["成交量"])
    ma15_now = df['收盘'].iloc[-15:].mean()
    price_max = df['收盘'].max()
    price_min = df['收盘'].min()

    price_position_pct = (ma15_now - price_min) / (price_max - price_min)

    obv_slope_15  = calculate_slope(df['obv'].iloc[-15:])
    ma60_slope_10 = calculate_slope(df['MA60'].iloc[-10:])

    df['golden_cross'] = (df['MA5'].shift(1) < df['MA20'].shift(1)) & (df['MA5'] >= df['MA20'])
    recent_cross = df['golden_cross'].iloc[-5:]

    # MA5 大于 MA60
    ma5_cross_ma60 = np.average(df['MA5'].iloc[-5:]) >= np.average(df['MA60'].iloc[-5:])

    # 综合条件判断
    if recent_cross.any() \
        and df['MA5'].iloc[-1] > df['MA20'].iloc[-1] \
        and obv_slope_15 > 0 \
        and ma60_slope_10 > 0 \
        and ma5_cross_ma60:

        print(f"✅ {code} {name} 符合所有选股条件，当前价位处于 {price_position_pct*100}%")
    else:
        pass


def main():
    # 拉取实时行情（包括市盈率）
#    spot_df = ak.stock_zh_a_spot_em()
#    columns = spot_df.columns.tolist()

    # Read from CSV file
    spot_df = pd.read_excel("/home/sun/data/other/stock_realtime_data.xlsx", dtype={"代码": str})
    columns = spot_df.columns.tolist()

    # 自动识别市盈率列名
    pe_candidates = ["市盈率-动态", "市盈率(动)", "市盈率", "市盈率_TTM"]
    pe_col = next((col for col in pe_candidates if col in columns), None)


    if not pe_col:
        raise ValueError("无法识别市盈率字段，请检查 AkShare 返回列：\n" + str(columns))

    # 把市盈率数据拉出来然后清除小于0的
    spot_df = spot_df[["代码", "名称", pe_col]].dropna()
    spot_df = spot_df[spot_df[pe_col] > 0]
    
    #print(spot_df)
    # 日期范围
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(days=365*5)).strftime("%Y%m%d")

    for index, row in spot_df.iterrows():
        stock_code = row["代码"]  # 假设“代码”是股票代码列名
        #print(stock_code, type(stock_code))
        check_stock_MA(stock_code, row["名称"], row[pe_col], start_date, end_date)

    #check_stock_MA("600012", "玉禾田", 11.8, start_date, end_date)



#    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="300815", period="daily", start_date=start_date, end_date=end_date, adjust="")
    

#
#    print(stock_zh_a_hist_df)

#    print(spot_df.iloc[0]["代码"], spot_df.iloc[0]["名称"], spot_df.iloc[0][pe_col], start_date, end_date)

#    check_stock_MA(spot_df.iloc[0]["代码"], spot_df.iloc[0]["名称"], spot_df.iloc[0][pe_col], start_date, end_date)
#    results = []
#    with ThreadPoolExecutor(max_workers=1) as executor:
#        futures = {
#            executor.submit(
#                check_stock_MA,
#                row["代码"],
#                row["名称"],
#                row[pe_col],
#                start_date,
#                end_date
#            ): row["代码"] for _, row in spot_df.iterrows()
#        }
#        for future in as_completed(futures):
#            res = future.result()
#            if res:
#                results.append(res)
#
#    if results:
#        print(f"\n✅ 满足条件的股票共 {len(results)} 只：")
#        for code, name in sorted(results):
#            print(f"{code}  {name}")
#    else:
#        print("❌ 当前无股票满足所有条件。")

if __name__ == "__main__":
    main()
