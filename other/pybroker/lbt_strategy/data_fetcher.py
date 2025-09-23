import akshare as ak
import pandas as pd
import os
import time
import config # Import config to check the benchmark ticker

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
    Fetches and caches historical OHLCV data.
    Smarter version: Uses the correct akshare function for stocks vs. indices.
    """
    cache_path = os.path.join('data', f'{ticker}.csv')
    if os.path.exists(cache_path):
        # Even if cached, check if the date range is sufficient
        df_cached = pd.read_csv(cache_path, index_col='date', parse_dates=True)
        if df_cached.index.min() <= pd.to_datetime(start_date) and df_cached.index.max() >= pd.to_datetime(end_date):
             return df_cached

    print(f"Fetching data for {ticker}...")
    try:
        start_str = start_date.replace('-', '')
        end_str = end_date.replace('-', '')
        df = pd.DataFrame()

        # --- THIS IS THE NEW, SMARTER LOGIC ---
        is_a_share_index = ticker.startswith(('sh00', 'sh95', 'sz39'))
        is_hk_index = ticker == 'hkHSI'

        if is_a_share_index:
            df = ak.index_zh_a_hist(symbol=ticker, period="daily", start_date=start_str, end_date=end_str)
        elif is_hk_index:
            # The symbol for Hang Seng in akshare is "HSI"
            df = ak.index_hk_hist(symbol="HSI", period="daily", start_date=start_str, end_date=end_str)
        elif ticker.startswith('hk'):
            # Hong Kong stocks
            df = ak.stock_hk_hist(symbol=ticker[2:], period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        else:
            # A-shares (default)
            df = ak.stock_zh_a_hist(symbol=ticker, period="daily", start_date=start_str, end_date=end_str, adjust="qfq")
        
        if df.empty:
            return None

        # Standardize column names
        df.rename(columns={'日期': 'date', '开盘': 'open', '收盘': 'close', '最高': 'high', '最低': 'low', '成交量': 'volume', '成交额': 'turnover'}, inplace=True, errors='ignore')
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        os.makedirs('data', exist_ok=True)
        df.to_csv(cache_path)
        time.sleep(0.5) # Be polite to the API server
        return df
    except Exception as e:
        print(f"Could not fetch data for {ticker}: {e}")
        return None
