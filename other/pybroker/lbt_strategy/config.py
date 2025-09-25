# --- Backtest Settings ---
START_DATE = '2022-01-01'
END_DATE = '2024-01-01'
INITIAL_CAPITAL = 1000000
COMMISSION_BPS = 10  # Basis points (e.g., 10 bps = 0.10%)
STOCK_UNIVERSE = 'A-shares'  # 'A-shares' or 'HK-shares'
BENCHMARK_TICKER = 'sh000300' # 'sh000300' for CSI 300, 'hkHSI' for Hang Seng

# --- Liquidity & Event Filters ---
MIN_AVG_TURNOVER = 50000000  # 50 million in currency units

# --- L-B-T Indicator Parameters ---
# Low Parameters
LOW_52_WEEK_PERCENTILE_THRESHOLD = 0.30 # Bottom 30% of 52-week range
LOW_SMA200_PROXIMITY_THRESHOLD = 1.05 # Price must be <= 5% above SMA200

# Base Parameters
BASE_BBW_THRESHOLD = 0.10 # Max Bollinger Bandwidth is 10% of middle band
BASE_ATR_PERCENT_THRESHOLD = 0.03 # Max ATR is 3% of the closing price
BASE_SMA200_SLOPE_DAYS = 60 # Lookback period for SMA200 slope
BASE_SMA200_MAX_SLOPE = 0.0002 # Max slope (0.02%/day) for a flat base

# Turn Parameters
TURN_BREAKOUT_LOOKBACK = 40 # Lookback period for price breakout (days)
TURN_VOLUME_MULTIPLIER = 1.5 # Breakout volume must be 1.5x average volume
TURN_MA_ALIGNMENT = {'fast': 20, 'medium': 50, 'slow': 200}

# --- Scoring Weights ---
WEIGHT_LOW = 0.40
WEIGHT_BASE = 0.35
WEIGHT_TURN = 0.25
SCORE_THRESHOLD = 0.75 # Minimum composite score to trigger a buy signal

# --- Portfolio & Risk Management ---
MAX_ACTIVE_POSITIONS = 10
POSITION_SIZING_METHOD = 'equal_weight' # 'equal_weight' or 'volatility_target'
STOP_LOSS_PCT = 0.08  # 8% stop-loss from entry price
PROFIT_TAKE_PCT = 0.20 # 20% profit target from entry price

# --- NEW: Daily Screener Settings ---
# A list of markets to scan every day.
SCREENER_UNIVERSE = ['A-shares', 'HK-shares'] 

# How many recent days to check for a buy signal.
SCREENER_LOOKBACK_DAYS = 30 

# The interval to pause between fetching data for each stock to avoid API blocks.
# A random value between these two numbers will be used.
SLEEP_INTERVAL_SECONDS = (5, 10) 