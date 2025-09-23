import pandas as pd
import numpy as np

def calculate_all_indicators(df, config):
    """Calculates all required technical indicators for a given dataframe."""
    for period in config.TURN_MA_ALIGNMENT.values():
        df[f'sma{period}'] = df['close'].rolling(window=period).mean()

    sma20 = df['sma20']
    std20 = df['close'].rolling(window=20).std()
    df['bb_upper'] = sma20 + (std20 * 2)
    df['bb_lower'] = sma20 - (std20 * 2)
    df['bbw'] = (df['bb_upper'] - df['bb_lower']) / sma20

    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    
    df['52w_high'] = df['close'].rolling(window=252).max()
    df['52w_low'] = df['close'].rolling(window=252).min()
    df['52w_percentile'] = (df['close'] - df['52w_low']) / (df['52w_high'] - df['52w_low'])
    
    slope_days = config.BASE_SMA200_SLOPE_DAYS
    df[f'sma200_slope'] = df['sma200'].diff(periods=slope_days) / slope_days
    
    return df