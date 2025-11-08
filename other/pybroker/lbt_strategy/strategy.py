import pandas as pd

def calculate_lbt_score(stock_data, config):
    """Calculates the Low, Base, and Turn scores for each day."""
    scores = pd.DataFrame(index=stock_data.index)
    
    low_cond1 = stock_data['52w_percentile'] <= config.LOW_52_WEEK_PERCENTILE_THRESHOLD
    low_cond2 = (stock_data['close'] / stock_data['sma200']) <= config.LOW_SMA200_PROXIMITY_THRESHOLD
    scores['low_score'] = (low_cond1 & low_cond2).astype(int)

    base_cond1 = stock_data['bbw'] <= config.BASE_BBW_THRESHOLD
    base_cond2 = stock_data['atr_pct'] <= config.BASE_ATR_PERCENT_THRESHOLD
    base_cond3 = abs(stock_data['sma200_slope'] / stock_data['sma200']) <= config.BASE_SMA200_MAX_SLOPE
    scores['base_score'] = (base_cond1 & base_cond2 & base_cond3).astype(int)

    ma_fast, ma_medium, ma_slow = config.TURN_MA_ALIGNMENT.values()
    
    turn_cond1 = stock_data['close'] > stock_data['close'].shift(1).rolling(window=config.TURN_BREAKOUT_LOOKBACK).max()
    turn_cond2 = stock_data['volume'] > (stock_data['avg_volume_20'] * config.TURN_VOLUME_MULTIPLIER)
    turn_cond3 = (stock_data[f'sma{ma_fast}'] > stock_data[f'sma{ma_medium}']) & \
                 (stock_data[f'sma{ma_medium}'] > stock_data[f'sma{ma_slow}'])
    
    scores['turn_score'] = (turn_cond1 & turn_cond2 & turn_cond3).astype(int)

    scores['composite_score'] = (scores['low_score'] * config.WEIGHT_LOW +
                                 scores['base_score'] * config.WEIGHT_BASE +
                                 scores['turn_score'] * config.WEIGHT_TURN)
                                 
    return scores