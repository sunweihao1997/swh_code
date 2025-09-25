import akshare as ak
import pandas as pd
import os
import time
from datetime import datetime

def get_stock_list(universe='A-shares'):
    """Fetches the list of all stocks for a given market."""
    print(f"Fetching stock list for {universe}...")
    if universe == 'A-shares':
        stock_df = ak.stock_zh_a_spot_em()
        return stock_df['代码'].tolist()
    elif universe == 'HK-shares':
        stock_df = ak.stock_hk_spot()
        return stock_df['代码'].tolist()
    return []

def fetch_stock_data(ticker, start_date, end_date):
    """
    Fetches and caches historical OHLCV data with smart caching.
    - If a file from today exists, load it from disk.
    - If the file is old or doesn't exist, fetch new data from the API.
    """
    cache_path = os.path.join('data', f'{ticker}.csv')
    today_date = datetime.now().date()

    # --- NEW SMART CACHING LOGIC ---
    if os.path.exists(cache_path):
        mod_time = os.path.getmtime(cache_path)
        mod_date = datetime.fromtimestamp(mod_time).date()
        if mod_date == today_date:
            print(f"Loading cached data for {ticker} (from today)...", end="")
            try:
                return pd.read_csv(cache_path, index_col='date', parse_dates=True)
            except Exception as e:
                print(f" Error loading cache for {ticker}: {e}. Will fetch fresh data.")

    # --- If cache is old or doesn't exist, proceed to fetch ---
    print(f"Fetching fresh data for {ticker}...", end="")
    try:
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        df = pd.DataFrame()

        is_a_share_index = ticker.startswith(('sh00', 'sh95', 'sz39'))
        is_hk_index = ticker == 'hkHSI'

        if is_a_share_index:
            df = ak.index_zh_a_hist(symbol=ticker, period="daily", start_date=start_str, end_date=end_str)
        elif is_hk_index:
            df = ak.index_hk_hist(symbol="HSI", period="daily", start_date=start_str, end_date=end_str)
        elif ticker.startswith('hk'):
            df = ak.stock_hk_hist(symbol=ticker[2:], period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        else:
            df = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        
        if df.empty:
            print(" No data returned from API.")
            return None

        # Standardize column names
        df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'turnover'}, inplace=True, errors='ignore')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        os.makedirs('data', exist_ok=True)
        df.to_csv(cache_path)
        time.sleep(0.5)
        return df
    except Exception as e:
        print(f" Could not fetch data for {ticker}: {e}")
        return None

