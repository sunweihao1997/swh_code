import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas_ta as ta
import numpy as np

def cal_index_for_stock(stock_num, start_date, end_date):
    """
    Calculate technical indicators for a given stock.
    """
    # Get historical data for the stock
    df = ak.stock_zh_a_hist(symbol=stock_num, start_date=start_date, end_date=end_date, adjust="qfq")
    print(f"Data length for {stock_num}: {len(df)}")
    if len(df) < 200:
        return None
    else:
        df.to_excel(f"/home/ubuntu/stock_data/stock_price_single/{stock_num}.xlsx", index=False)
        
        # Calculate CCI
        df['CCI'] = ta.cci(high=df['最高'], low=df['最低'], close=df['收盘'], length=14)
        
        # Calculate MACD
        #print(df['收盘'])
        macd = ta.macd(df['收盘'])
        df = pd.concat([df, macd], axis=1)
        
        # Calculate OBV
        #print(df[['收盘', '成交量']].head(40))
        df['OBV'] = ta.obv(close=df['收盘'], volume=df['成交量'])
        df['OBVM30'] = df['OBV'].rolling(window=30).mean()
    
        # Calculate TRIX
        df['trix'] = ta.trix(df['收盘'], length=12).iloc[:, 0]
    
        df['trix_signal'] = df['trix'].rolling(window=9).mean()
        df['trix_diff'] = df['trix'] - df['trix_signal']
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['收盘'], length=6)

        #additional indicators
        # ------------ 1.  recent 10 days slope for CCI ------------------
        #print(df['CCI'].tail(10))
        df['CCI_slope'] = df['CCI'].rolling(window=10).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        df['CCI_slope20'] = df['CCI'].rolling(window=20).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        #print(df['CCI_slope'].tail(10)) # Have checked and results are correct

        # ------------ 2.  recent ratio of positive CCI among past 14 days -----------------
        N_window = 10 ; threshold_cci = 0.6
        cci_pos = (df['CCI'] > 0).astype(int) ; cci_neg = (df['CCI'] < 0).astype(int)
        cci_pos_ratio = cci_pos.rolling(window=N_window).sum() / N_window

        # ------------ 3.  Slope of OBV index -----------------------
        df['OBV_slope'] = df['OBV'].rolling(window=7).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        df['OBV_slope20'] = df['OBV'].rolling(window=20).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        # ------------ 4.  Ratio inwhich OBV larger than OBV30; using 10 days -----------------
        df['OBV_M30']   = df['OBV'].rolling(window=30).mean()
        obv_pos           = ((df['OBV'] - df['OBV_M30']) > 0).astype(int)
        obv_pos_ratio = obv_pos.rolling(window=10).sum() / 10
        df['OBV_ratio'] = obv_pos_ratio
        #print(obv_pos_ratio)

        # ------------ 5. Slope of DIFF_MACD ------------
        df['macd_diff_slope'] = df['MACDh_12_26_9'].rolling(window=5).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        df['macd_diff_slope20'] = df['MACDh_12_26_9'].rolling(window=20).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        # ------------ 6. Slope of MA60 --------------
        df['MA60'] = df['收盘'].rolling(window=55).mean()
        df['MA60_slope'] = df['MA60'].rolling(window=3).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )

        # ------------ 6. Slope of trix-diff --------------
        df['trix_diff_slope'] = df['trix_diff'].rolling(window=5).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
    
    
        return df

def cal_index_for_stock_hk(stock_num, start_date, end_date):
    """
    Calculate technical indicators for a given stock.
    """
    # Get historical data for the stock
    df = ak.stock_hk_hist(symbol=stock_num, period="daily", start_date=start_date, end_date=end_date, adjust="qfq")
    if len(df) < 200:
        return None
    else:
        df.to_excel(f"/home/ubuntu/stock_data/stock_price_single/HK_{stock_num}.xlsx", index=False)
        
        # Calculate CCI
        df['CCI'] = ta.cci(high=df['最高'], low=df['最低'], close=df['收盘'], length=14)
        
        # Calculate MACD
        #print(df['收盘'])
        macd = ta.macd(df['收盘'])
        df = pd.concat([df, macd], axis=1)
        
        # Calculate OBV
        #print(df[['收盘', '成交量']].head(40))
        df['OBV'] = ta.obv(close=df['收盘'], volume=df['成交量'])
        df['OBVM30'] = df['OBV'].rolling(window=30).mean()
    
        # Calculate TRIX
        df['trix'] = ta.trix(df['收盘'], length=12).iloc[:, 0]
    
        df['trix_signal'] = df['trix'].rolling(window=9).mean()
        df['trix_diff'] = df['trix'] - df['trix_signal']
        
        # Calculate RSI
        df['rsi'] = ta.rsi(df['收盘'], length=6)

        #additional indicators
        # ------------ 1.  recent 10 days slope for CCI ------------------
        #print(df['CCI'].tail(10))
        df['CCI_slope'] = df['CCI'].rolling(window=10).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        df['CCI_slope20'] = df['CCI'].rolling(window=20).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        #print(df['CCI_slope'].tail(10)) # Have checked and results are correct

        # ------------ 2.  recent ratio of positive CCI among past 14 days -----------------
        N_window = 10 ; threshold_cci = 0.6
        cci_pos = (df['CCI'] > 0).astype(int) ; cci_neg = (df['CCI'] < 0).astype(int)
        cci_pos_ratio = cci_pos.rolling(window=N_window).sum() / N_window

        # ------------ 3.  Slope of OBV index -----------------------
        df['OBV_slope'] = df['OBV'].rolling(window=7).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        df['OBV_slope20'] = df['OBV'].rolling(window=20).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        # ------------ 4.  Ratio inwhich OBV larger than OBV30; using 10 days -----------------
        df['OBV_M30']   = df['OBV'].rolling(window=30).mean()
        obv_pos           = ((df['OBV'] - df['OBV_M30']) > 0).astype(int)
        obv_pos_ratio = obv_pos.rolling(window=10).sum() / 10
        df['OBV_ratio'] = obv_pos_ratio
        #print(obv_pos_ratio)

        # ------------ 5. Slope of DIFF_MACD ------------
        df['macd_diff_slope'] = df['MACDh_12_26_9'].rolling(window=5).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        df['macd_diff_slope20'] = df['MACDh_12_26_9'].rolling(window=20).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
        # ------------ 6. Slope of MA60 --------------
        df['MA60'] = df['收盘'].rolling(window=55).mean()
        df['MA60_slope'] = df['MA60'].rolling(window=3).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )

        # ------------ 6. Slope of trix-diff --------------
        df['trix_diff_slope'] = df['trix_diff'].rolling(window=5).apply(
            lambda x: LinearRegression().fit(np.arange(len(x)).reshape(-1, 1), x).coef_[0],
            raw=True
        )
    
    
        return df


def standardize_and_normalize(df):
    """
    对输入 DataFrame 中的各列进行标准化/归一化。
    
    输入 df 需要包含:
    ['股票代码', '日期', '收盘', '成交量', 'CCI', 'MACD_12_26_9', 'MACDs_12_26_9', 
    'MACDh_12_26_9', 'OBV', 'trix', 'trix_signal', 'trix_diff', 'rsi']
    """

    # 初始化参数保存字典
    stats_dict = {}

    # 对每只股票分别处理
    for stock, group in df.groupby('股票代码'):

        # 复制 group，避免 SettingWithCopyWarning
        g = group.copy()

        # ========== 收盘标准化 ==========
        close_mean = g['收盘'].mean()
        close_std = g['收盘'].std()
        g['收盘_z'] = (g['收盘'] - close_mean) / close_std

        # ========== 成交量标准化 ==========
        vol_mean = g['成交量'].mean()
        vol_std = g['成交量'].std()
        g['成交量_z'] = (g['成交量'] - vol_mean) / vol_std

        # ========== RSI 归一化 (0-1) ==========
        g['rsi_norm'] = g['rsi'] / 100

        # ========== OBV 变化率 + 标准化 ==========
        g['obv_pct'] = g['OBV'].pct_change()
        obv_pct_mean = g['obv_pct'].mean()
        obv_pct_std = g['obv_pct'].std()
        g['obv_pct_z'] = (g['obv_pct'] - obv_pct_mean) / obv_pct_std

        # ========== 其他指标标准化 ==========
        for col in ['CCI', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'trix', 'trix_signal', 'trix_diff', 'CCI_slope', 'OBV_slope', 'OBV_ratio', 'macd_diff_slope', 'MA60_slope', 'trix_diff_slope', 'macd_diff_slope20', 'CCI_slope20', 'OBV_slope20']:
            col_mean = g[col].mean()
            col_std = g[col].std()
            g[col + '_z'] = (g[col] - col_mean) / col_std

        # 保存均值与标准差，用于测试集/实盘
        stats_dict[stock] = {
            '收盘_mean': close_mean, '收盘_std': close_std,
            '成交量_mean': vol_mean, '成交量_std': vol_std,
            'obv_pct_mean': obv_pct_mean, 'obv_pct_std': obv_pct_std,
        }
        for col in ['CCI', 'MACD_12_26_9', 'MACDs_12_26_9', 'MACDh_12_26_9', 'trix', 'trix_signal', 'trix_diff']:
            stats_dict[stock][col + '_mean'] = g[col].mean()
            stats_dict[stock][col + '_std'] = g[col].std()

        # 将处理后的 group 替换到原 df
        df.loc[g.index, g.columns] = g

    return df, stats_dict

def map_df(df):
    # 定义中文列名到英文的映射字典，根据你的实际DataFrame修改补充
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

    # 如果你想保存替换后的 DataFrame 查看
    return df
