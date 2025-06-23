import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
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


def check_stock(code: str, name: str, pe: float, start_date: str, end_date: str) -> tuple | None:

    if pe >= 25 or pe <5:
        #print(f"{code} {name} 被排除：PE >= 30 ({pe:.2f})")
        return None
    else:
        df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
        print(df)
        #print(f"Checking {code} {name} with PE {pe:.2f}")
        #return (code, name)
    
#        try:
#            df = ak.stock_zh_a_hist(symbol=code, start_date=start_date, end_date=end_date, adjust="qfq")
#            if df.empty or len(df) < 130:
#                print("Pass")
#                return None
#        except Exception as e:
#            print(f"抓取失败：{code}，错误信息：{e}")
#            return None
#    
#        df["MA60"] = df["收盘"].rolling(60).mean()
#        df["MA120"] = df["收盘"].rolling(120).mean()
#        latest = df.iloc[-1]
#        print("Complete calculating moving averages for", code, name)
#
#    # 条件判断
##    if latest["收盘"] <= latest["MA60"]:
##        return None
#    print(f"Checking {code} {name} with PE {pe:.2f}")
#    if latest["MA60"] <= latest["MA120"]:
#        return None
#    if not slope_upward_regression(df["MA60"], 5):
#        return None
##    if not slope_upward_regression(df["MA120"], 5):
##        return None

#    return (code, name)
#def check_stock(code, name, pe, start_date, end_date):
#    if pe <= 0:
#        print(f"{code} {name} 被排除：PE <= 0")
#        return None
#    elif pe >= 30:
#        print(f"{code} {name} 被排除：PE >= 30 ({pe:.2f})")
#        return None
#
#    return (code, name)


def main():
    # 拉取实时行情（包括市盈率）
    spot_df = ak.stock_zh_a_spot_em()
    columns = spot_df.columns.tolist()

    # 自动识别市盈率列名
    pe_candidates = ["市盈率-动态", "市盈率(动)", "市盈率", "市盈率_TTM"]
    pe_col = next((col for col in pe_candidates if col in columns), None)


    if not pe_col:
        raise ValueError("无法识别市盈率字段，请检查 AkShare 返回列：\n" + str(columns))

    # 把市盈率数据拉出来然后清除小于0的
    spot_df = spot_df[["代码", "名称", pe_col]].dropna()
    spot_df = spot_df[spot_df[pe_col] > 0]
    

    # 日期范围
    end_date = datetime.today().strftime("%Y%m%d")
    start_date = (datetime.today() - timedelta(days=300)).strftime("%Y%m%d")
    stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="300815", period="daily", start_date="20170301", end_date='20240528', adjust="")

    #print(spot_df.iloc[0]["代码"], spot_df.iloc[0]["名称"], spot_df.iloc[0][pe_col], start_date, end_date)

#    check_stock(spot_df.iloc[0]["代码"], spot_df.iloc[0]["名称"], spot_df.iloc[0][pe_col], start_date, end_date)
#    results = []
#    with ThreadPoolExecutor(max_workers=1) as executor:
#        futures = {
#            executor.submit(
#                check_stock,
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
