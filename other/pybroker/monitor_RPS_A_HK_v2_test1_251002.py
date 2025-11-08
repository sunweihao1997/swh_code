import pandas as pd
import numpy as np
import akshare as ak
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def get_close_col(df):
    # 兼容不同接口的列名
    candidates = ['收盘', '收盘价', 'close', 'Close']
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"未找到收盘价列，现有列：{list(df.columns)[:10]}")

def get_date_col(df):
    candidates = ['日期', 'date', 'Date', '时间', '交易日期']
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"未找到日期列，现有列：{list(df.columns)[:10]}")

def fetch_hist_sanitized(symbol, market_type='A'):
    end_date = pd.Timestamp.today().strftime('%Y%m%d')
    start_date = (pd.Timestamp.today() - pd.Timedelta(days=420)).strftime('%Y%m%d')
    if market_type == 'A':
        df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    else:
        # 先别跑港股，确认 A 股逻辑 OK 再说
        raise NotImplementedError("先验证 A 股，HK 稍后再开")
    # 列名与类型处理
    dcol = get_date_col(df)
    ccol = get_close_col(df)
    df = df[[dcol, ccol]].copy()
    df[dcol] = pd.to_datetime(df[dcol])
    df[ccol] = pd.to_numeric(df[ccol], errors='coerce')
    df = df.sort_values(dcol).dropna()
    # 至少 150MA + 余量，且我们后面会用到 iloc[-1]
    if len(df) < 160:
        return pd.DataFrame()
    # 计算 MA
    df['MA5'] = df[ccol].rolling(window=5, min_periods=5).mean()
    df['MA60'] = df[ccol].rolling(window=60, min_periods=60).mean()
    df['MA150'] = df[ccol].rolling(window=150, min_periods=150).mean()
    df = df.dropna()
    return df

def quick_screen_A():
    ok = []
    fails = 0
    all_list = ak.stock_zh_a_spot_em()
    # 为了快，先只取成交额 Top 500 作为候选池
    key = '成交额' if '成交额' in all_list.columns else None
    if key:
        all_list = all_list.nlargest(500, key)
    for row in all_list.itertuples(index=False):
        code = getattr(row, '代码')
        try:
            hist = fetch_hist_sanitized(code, market_type='A')
            if hist.empty:
                fails += 1
                continue
            latest = hist.iloc[-1]
            if latest['MA5'] > latest['MA60'] and latest['MA60'] > latest['MA150']:
                ok.append(code)
        except Exception as e:
            fails += 1
            logging.warning(f"A-{code} 异常: {e}")
            continue

        
    logging.info(f"候选数: {len(all_list)}, 命中: {len(ok)}, 失败/跳过: {fails}")
    return ok

if __name__ == "__main__":
    winners = quick_screen_A()
    print("满足 MA5>MA60>MA150 的 A 股数量：", len(winners))
    print(winners[:30])
    #stock_zh_a_hist_df = ak.stock_zh_a_hist(symbol="000001", period="daily", start_date="20170301", end_date='20240528', adjust="")
    #print(stock_zh_a_hist_df.head(3))
