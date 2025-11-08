# -*- coding: utf-8 -*-
"""
This script integrates three separate analysis components into a single VCP strategy:
1.  Detecting periods of low volatility (ATR/BBW shrinking).
2.  Identifying VCP patterns with a relaxed condition for lows.
3.  Pinpointing a high-volume breakout as a potential buy signal.
4.  Running a backtest to measure the performance of breakout signals.
5.  Visualizing the entire pattern on a candlestick chart.
"""
import akshare as ak
import pandas as pd
import pandas_ta as ta
import numpy as np
import mplfinance as mpf
from typing import List, Tuple, Optional, Dict
import time # Import the time module for sleep
import random # Import the random module for variable sleep
import os # To handle file paths and directories
import matplotlib.pyplot as plt # To save and close figures

# =============================================================================
# SECTION 0: DATA FETCHING & UTILITIES (REVISED LOGIC)
# =============================================================================

def get_stock_data(stock_code: str, period: str = "daily", start_date: str = "20200101", end_date: str = "20251231") -> Optional[pd.DataFrame]:
    """
    Fetches historical stock data using Akshare. It first tries to fetch data as a
    Chinese A-share. If that returns no data, it then tries to fetch it as a
    Hong Kong stock.
    Standardizes column names to lowercase ('open', 'high', 'low', 'close', 'volume').
    """
    stock_df = pd.DataFrame()
    # print(f"Fetching data for stock: {stock_code}...") # Quieter for screening

    # --- UPDATED LOGIC ---
    # First, attempt to fetch as a Chinese A-share stock.
    try:
        # print("Attempting to fetch as a Chinese A-share...")
        stock_df = ak.stock_zh_a_hist(symbol=stock_code, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception as e:
        # print(f"An error occurred while fetching A-share data: {e}")
        # If there's an error, we can proceed to try the HK market.
        pass

    # If the A-share fetch returned an empty DataFrame, try as a Hong Kong stock.
    if stock_df.empty:
        try:
            # print("A-share data not found or empty. Attempting to fetch as a Hong Kong stock...")
            stock_df = ak.stock_hk_hist(symbol=stock_code, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
        except Exception as e:
            # print(f"Could not fetch data for {stock_code} from either market. Final error: {e}")
            return None

    if stock_df.empty:
        # print(f"No data returned for {stock_code} from either market.")
        return None
        
    # Standardize column names to lowercase
    stock_df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}, inplace=True)
    
    # Ensure standard columns exist
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in stock_df.columns for col in required_cols):
        # print(f"Data for {stock_code} is missing required columns.")
        return None
        
    stock_df = stock_df[required_cols]
    
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.set_index('date', inplace=True)
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
        
    stock_df.dropna(inplace=True)
    
    # print("Data fetched successfully.")
    return stock_df.sort_index()


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples daily data to weekly, ending on Friday."""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample('W-FRI').agg(agg).dropna(how='any')


# =============================================================================
# SECTION 1: STAGE 1 - DETECT LOW VOLATILITY EXECUTION WINDOW
# From your Script 1
# =============================================================================

def percentile_rank_last(series: pd.Series, window: int) -> pd.Series:
    """Calculates the percentile rank of the last value in a rolling window."""
    def _rank(x):
        s = pd.Series(x).dropna()
        if len(s) == 0: return np.nan
        return s.rank(pct=True).iloc[-1]
    return series.rolling(window, min_periods=max(5, window // 5)).apply(_rank, raw=False)

def find_execution_windows(df_daily: pd.DataFrame, exec_window_days: int = 40) -> pd.Series:
    """
    Finds candidate periods for VCP analysis. An execution window opens after a week
    of low volatility, and we then look for a day of low volatility inside that window.
    Returns a boolean Series indicating active execution windows.
    """
    # --- Daily Calculations ---
    d = df_daily.copy()
    d['ATR20_d'] = ta.atr(high=d['high'], low=d['low'], close=d['close'], length=20)
    d['BBW20_d'] = ta.bbands(close=d['close'], length=20, std=2.0).iloc[:, 3] # BBB_20_2.0
    d['rank_ATR_d'] = percentile_rank_last(d['ATR20_d'], window=252)
    d['rank_BBW_d'] = percentile_rank_last(d['BBW20_d'], window=252)
    d['daily_low_vol'] = (d[['rank_ATR_d', 'rank_BBW_d']].min(axis=1) <= 0.20)

    # --- Weekly Calculations ---
    w = to_weekly(df_daily)
    w['ATR20_w'] = ta.atr(high=w['high'], low=w['low'], close=w['close'], length=20)
    w['BBW20_w'] = ta.bbands(close=w['close'], length=20, std=2.0).iloc[:, 3]
    w['rank_ATR_w'] = percentile_rank_last(w['ATR20_w'], window=52)
    w['rank_BBW_w'] = percentile_rank_last(w['BBW20_w'], window=52)
    w['weekly_low_vol'] = (w[['rank_ATR_w', 'rank_BBW_w']].min(axis=1) <= 0.35)

    # --- Link Weekly to Daily ---
    is_window_open = pd.Series(False, index=df_daily.index)
    for dt, is_low_vol_week in w[w['weekly_low_vol']].iterrows():
        start_date = dt + pd.Timedelta(days=1)
        end_date = start_date + pd.Timedelta(days=exec_window_days)
        is_window_open.loc[start_date:end_date] = True

    candidates = is_window_open & d['daily_low_vol'].reindex(df_daily.index).fillna(False)
    return candidates

# =============================================================================
# SECTION 2: STAGE 2 - VCP PATTERN DETECTION (WITH RELAXED CONDITION)
# =============================================================================
ContractionPairs = List[Tuple[pd.Timestamp, float, pd.Timestamp, float, float]]

def find_swings_zigzag(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Identifies swing highs and lows using a ZigZag logic without lookahead."""
    price_h, price_l = df["high"], df["low"]
    n = len(df)
    swing_high = np.full(n, np.nan, dtype=float)
    swing_low  = np.full(n, np.nan, dtype=float)
    
    direction = 0
    max_price, min_price = price_h.iloc[0], price_l.iloc[0]
    max_i, min_i = 0, 0

    for i in range(1, n):
        hi, lo = price_h.iloc[i], price_l.iloc[i]
        
        if direction == 0:
            if hi > max_price: max_price, max_i = hi, i
            if lo < min_price: min_price, min_i = lo, i
            if (hi - min_price) / min_price >= threshold and min_i < i:
                swing_low[min_i] = min_price
                direction, max_price, max_i = 1, hi, i
            elif (max_price - lo) / max_price >= threshold and max_i < i:
                swing_high[max_i] = max_price
                direction, min_price, min_i = -1, lo, i
        elif direction == 1: # Uptrend
            if hi > max_price: max_price, max_i = hi, i
            elif (max_price - lo) / max_price >= threshold and max_i < i:
                swing_high[max_i] = max_price
                direction, min_price, min_i = -1, lo, i
        elif direction == -1: # Downtrend
            if lo < min_price: min_price, min_i = lo, i
            elif (hi - min_price) / min_price >= threshold and min_i < i:
                swing_low[min_i] = min_price
                direction, max_price, max_i = 1, hi, i
                
    return pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low}, index=df.index)

def extract_contractions(swings: pd.DataFrame, min_drop: float = 0.03, max_drop: float = 0.40, min_bars: int = 5) -> ContractionPairs:
    """Pairs swing highs with subsequent swing lows to form contractions."""
    pairs = []
    
    sh = swings['swing_high'].dropna()
    sl = swings['swing_low'].dropna()
    
    for t_hi, v_hi in sh.items():
        # Find the first swing low after this high
        possible_lows = sl[sl.index > t_hi]
        if not possible_lows.empty:
            t_lo, v_lo = possible_lows.index[0], possible_lows.iloc[0]
            
            # Ensure this low isn't already paired with a closer high
            prev_highs = sh[sh.index < t_lo]
            if not prev_highs.empty and prev_highs.index[-1] != t_hi:
                continue
                
            if (t_lo - t_hi).days < min_bars: continue
            
            drop = (v_hi - v_lo) / v_hi
            if min_drop <= drop <= max_drop:
                pairs.append((t_hi, v_hi, t_lo, v_lo, drop))
                
    return pairs

def is_valid_vcp_relaxed(pairs: ContractionPairs, need_n: int = 2) -> Optional[Dict]:
    """
    Validates a VCP sequence with the RELAXED condition for higher lows.
    1. Finds the last 'need_n' contractions.
    2. Checks for decreasing contraction depth.
    3. Checks if each low is higher than the FIRST low in the sequence.
    """
    if len(pairs) < need_n: return None

    sub_pairs = pairs[-need_n:]
    drops = [p[4] for p in sub_pairs]
    lows = [p[3] for p in sub_pairs]
    
    # 1. Check for decreasing drops
    for i in range(len(drops) - 1):
        if drops[i] <= drops[i+1]:
            return None # Not a valid contraction sequence

    # 2. **RELAXED** Check for higher lows
    first_low = lows[0]
    for i in range(1, len(lows)):
        # Each subsequent low must be higher than the very first low.
        if lows[i] <= first_low:
            return None # Lows are not trending up relative to the start

    # This is a valid VCP pattern
    pivot_time = sub_pairs[-1][0]
    pivot_price = sub_pairs[-1][1]

    return {"valid": True, "pairs": sub_pairs, "pivot_time": pivot_time, "pivot_price": pivot_price}

# =============================================================================
# SECTION 3: STAGE 3 - BREAKOUT DETECTION
# =============================================================================

def find_breakout(df: pd.DataFrame, pivot_time: pd.Timestamp, pivot_price: float, look_ahead: int = 20) -> Optional[pd.Series]:
    """Looks for a breakout signal after the VCP completes."""
    scan_start = pivot_time + pd.Timedelta(days=1)
    scan_end = scan_start + pd.Timedelta(days=look_ahead)
    scan_df = df.loc[scan_start:scan_end]

    if scan_df.empty: return None

    # Price breakout condition
    price_break = scan_df['high'] > pivot_price

    # Volume breakout condition (daily volume > 1.8 * 50-day SMA)
    vol_sma50 = df['volume'].rolling(50, min_periods=20).mean()
    volume_break = scan_df['volume'] > 1.8 * vol_sma50.reindex(scan_df.index)
    
    breakouts = scan_df[price_break & volume_break]
    
    if not breakouts.empty:
        # Return the first breakout day
        return breakouts.iloc[0]
        
    return None

# =============================================================================
# SECTION 4: VISUALIZATION
# =============================================================================

def plot_vcp_analysis(df: pd.DataFrame, vcp_result: Dict, breakout_signal: Optional[pd.Series], stock_code: str, save_path: str):
    """
    Plots the candlestick chart with VCP contractions, pivot, and breakout point,
    and saves the figure to the specified path.
    """
    vcp_pairs = vcp_result['pairs']
    pivot_price = vcp_result['pivot_price']
    
    # Define plot range
    start_date = vcp_pairs[0][0] - pd.Timedelta(days=30)
    if breakout_signal is not None:
        end_date = breakout_signal.name + pd.Timedelta(days=15)
    else:
        end_date = df.index[-1]
    
    plot_df = df.loc[start_date:end_date]
    
    # Prepare lines for VCP contractions
    contraction_lines = []
    for p in vcp_pairs:
        line = [(p[0], p[1]), (p[2], p[3])]
        contraction_lines.append(line)
        
    # Prepare line for pivot resistance
    pivot_start_date = vcp_pairs[-1][0]
    pivot_end_date = breakout_signal.name if breakout_signal is not None else plot_df.index[-1]
    pivot_line = [(pivot_start_date, pivot_price), (pivot_end_date, pivot_price)]
    
    all_lines = contraction_lines + [pivot_line]
    line_colors = ['blue'] * len(contraction_lines) + ['red']
    
    # Prepare marker for breakout point
    add_plots = []
    if breakout_signal is not None:
        buy_marker_df = pd.DataFrame(index=plot_df.index)
        buy_marker_df['signal'] = np.nan
        buy_marker_df.loc[breakout_signal.name, 'signal'] = breakout_signal['low'] * 0.98
        
        buy_plot = mpf.make_addplot(buy_marker_df['signal'], type='scatter', marker='^', color='green', markersize=200)
        add_plots.append(buy_plot)

    # Plot the chart
    fig, axlist = mpf.plot(
        plot_df,
        type='candle',
        style='yahoo',
        title=f'VCP Analysis for {stock_code}',
        ylabel='Price',
        volume=True,
        ylabel_lower='Volume',
        figsize=(18, 10),
        alines=dict(alines=all_lines, colors=line_colors, linewidths=1.2),
        addplot=add_plots,
        returnfig=True
    )
    
    # Add text labels for contraction lows
    ax = axlist[0]
    for i, p in enumerate(vcp_pairs):
         ax.text(p[2], p[3], f' L{i+1}', verticalalignment='top', fontsize=9, color='black')
    
    # Save the figure to the specified path and close it to free memory
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    
# =============================================================================
# SECTION 5: BACKTESTING ENGINE (NEW)
# =============================================================================
def run_backtest(df: pd.DataFrame, breakout_signal: pd.Series) -> Dict:
    """
    Runs a backtest on a breakout signal to calculate performance metrics.
    """
    backtest_results = {}
    
    breakout_date = breakout_signal.name
    entry_price = breakout_signal['close']
    
    # 1. Fixed-Period Holding Returns
    holding_periods = {'20d_ret': 20, '60d_ret': 60}
    for name, days in holding_periods.items():
        exit_date_loc = df.index.get_loc(breakout_date) + days
        if exit_date_loc < len(df.index):
            exit_price = df.iloc[exit_date_loc]['close']
            ret = (exit_price - entry_price) / entry_price
            backtest_results[name] = f"{ret:.2%}"
        else:
            backtest_results[name] = "N/A"
            
    # 2. Strategy-Based Return (Stop-Loss / Take-Profit)
    stop_loss_pct = -0.08
    take_profit_pct = 0.25
    max_hold_days = 120 # Maximum days to hold before exiting
    
    stop_loss_price = entry_price * (1 + stop_loss_pct)
    take_profit_price = entry_price * (1 + take_profit_pct)
    
    trade_df = df.loc[df.index > breakout_date].iloc[:max_hold_days]
    
    exit_reason = "Max Hold"
    exit_price = trade_df.iloc[-1]['close'] if not trade_df.empty else entry_price
    days_held = len(trade_df)

    for i, row in trade_df.iterrows():
        if row['low'] <= stop_loss_price:
            exit_reason = "Stop-Loss"
            exit_price = stop_loss_price
            days_held = (i - breakout_date).days
            break
        if row['high'] >= take_profit_price:
            exit_reason = "Take-Profit"
            exit_price = take_profit_price
            days_held = (i - breakout_date).days
            break
            
    strategy_ret = (exit_price - entry_price) / entry_price
    backtest_results['strat_ret'] = f"{strategy_ret:.2%}"
    backtest_results['strat_exit_reason'] = exit_reason
    backtest_results['strat_days_held'] = days_held
    
    return backtest_results

# =============================================================================
# SECTION 6: FULL MARKET SCREENING WORKFLOW (MODIFIED)
# =============================================================================
def screen_all_a_shares(start_date="20230101", end_date="20251231"):
    """
    Screens all Chinese A-share stocks for the VCP + Breakout pattern.
    Saves all successful matches and prints a summary at the end.
    """
    print("Fetching the list of all A-share stocks. This may take a moment...")
    try:
        all_stocks_df = ak.stock_zh_a_spot_em()
        print(f"Successfully fetched {len(all_stocks_df)} stock codes. Starting screening process...")
    except Exception as e:
        print(f"Failed to fetch stock list: {e}")
        return

    # Create a directory for charts if it doesn't exist
    chart_dir = "vcp_charts"
    os.makedirs(chart_dir, exist_ok=True)

    results_list = []
    total_stocks = len(all_stocks_df)

    # Iterate through the DataFrame to get both code and name
    for i, stock_row in all_stocks_df.iterrows():
        stock_code = stock_row['代码']
        stock_name = stock_row['名称']
        
        print(f"\n[{i+1}/{total_stocks}] Analyzing: {stock_code} ({stock_name})")
        
        # --- Define sleep duration ---
        sleep_time = random.uniform(5, 10)
        
        try:
            # --- STAGE 1: GET DATA ---
            df = get_stock_data(stock_code, start_date=start_date, end_date=end_date)
            if df is None or len(df) < 200:
                print(f"Skipping {stock_code}: Insufficient data.")
                time.sleep(sleep_time)
                continue

            # --- STAGE 2: FIND EXECUTION WINDOW ---
            candidate_days = find_execution_windows(df)
            if not candidate_days.any():
                print(f"Skipping {stock_code}: No low volatility window found.")
                time.sleep(sleep_time)
                continue
            
            last_candidate_date = candidate_days[candidate_days].index[-1]
            analysis_start_date = last_candidate_date - pd.Timedelta(days=120)
            analysis_df = df.loc[analysis_start_date:last_candidate_date]

            # --- STAGE 3: FIND VCP PATTERN ---
            swings = find_swings_zigzag(analysis_df, threshold=0.05)
            contractions = extract_contractions(swings)
            vcp_result = is_valid_vcp_relaxed(contractions, need_n=2)
            if not vcp_result:
                print(f"Skipping {stock_code}: No valid VCP pattern found.")
                time.sleep(sleep_time)
                continue
                
            # --- STAGE 4: DETECT BREAKOUT ---
            breakout_signal = find_breakout(df, vcp_result['pivot_time'], vcp_result['pivot_price'])
            if breakout_signal is None:
                print(f"Found VCP for {stock_code}, but no breakout yet.")
                time.sleep(sleep_time)
                continue

            # --- STAGE 5: RUN BACKTEST ---
            backtest_metrics = run_backtest(df, breakout_signal)

            # --- SUCCESS! SAVE RESULT ---
            print("="*50)
            print(f"SUCCESS! Found a complete VCP + Breakout pattern for stock: {stock_code} ({stock_name})")
            print(f"Breakout detected on {breakout_signal.name.date()}.")
            print("="*50)
            
            result_data = {
                "Stock Code": stock_code,
                "Stock Name": stock_name,
                "Pivot Time": vcp_result['pivot_time'].date(),
                "Pivot Price": f"{vcp_result['pivot_price']:.2f}",
                "Breakout Date": breakout_signal.name.date(),
                "Breakout Close": f"{breakout_signal['close']:.2f}",
                "Contraction Depths": ", ".join([f"{p[4]*100:.1f}%" for p in vcp_result['pairs']])
            }
            # Add backtest results to the dictionary
            result_data.update(backtest_metrics)
            results_list.append(result_data)
            
            # --- SAVE THE PLOT for every successful result ---
            breakout_date_str = breakout_signal.name.strftime('%Y-%m-%d')
            # Sanitize stock name for filename
            safe_stock_name = "".join([c for c in stock_name if c.isalnum()]).rstrip()
            file_name = f"{stock_code}_{safe_stock_name}_{breakout_date_str}.png"
            save_path = os.path.join(chart_dir, file_name)
            
            print(f"Saving chart to {save_path}...")
            plot_vcp_analysis(df, vcp_result, breakout_signal, f"{stock_code} - {stock_name}", save_path)


        except Exception as e:
            print(f"An error occurred while processing {stock_code}: {e}")
            time.sleep(sleep_time)
            continue
            
        print(f"Sleeping for {sleep_time:.1f} seconds...")
        time.sleep(sleep_time)
            
    # --- FINAL REPORT ---
    print("\n\n" + "="*80)
    print("Screening and Backtest complete.")
    if not results_list:
        print("No stocks with a full VCP + Breakout pattern were found.")
    else:
        results_df = pd.DataFrame(results_list)
        
        # Calculate and print summary statistics
        # Convert return columns to numeric for calculation
        for col in ['20d_ret', '60d_ret', 'strat_ret']:
             results_df[col] = pd.to_numeric(results_df[col].str.rstrip('%'), errors='coerce') / 100
        
        print(f"Found {len(results_df)} stocks matching the criteria. Backtest summary:")
        print(f"Average 20-Day Return: {results_df['20d_ret'].mean():.2%}")
        print(f"Average 60-Day Return: {results_df['60d_ret'].mean():.2%}")
        print(f"Average Strategy Return: {results_df['strat_ret'].mean():.2%}")
        print(f"Strategy Win Rate: {len(results_df[results_df['strat_ret'] > 0]) / len(results_df):.2%}")
        print("\nFull Results:")
        print(results_df.to_string())
        
        # Save to CSV
        try:
            csv_filename = 'vcp_breakout_results.csv'
            # Format percentage columns for CSV
            for col in ['20d_ret', '60d_ret', 'strat_ret']:
                results_df[col] = results_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
            results_df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
            print(f"\nResults have been successfully saved to {csv_filename}")
        except Exception as e:
            print(f"\nError saving results to CSV: {e}")
    print("="*80)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the desired analysis.
    Set to run the full market screener by default.
    """
    screen_all_a_shares()

    # --- To test a single stock, comment out the line above and uncomment the block below ---
    # print("--- Running single stock analysis ---")
    # stock_to_test = "002594"  # BYD Company (A-Share)
    # start_date = "20230101"
    # end_date = "20251231"
    
    # df = get_stock_data(stock_to_test, start_date=start_date, end_date=end_date)
    # if df is None or len(df) < 200:
    #     print("Not enough data to perform analysis.")
    #     return

    # candidate_days = find_execution_windows(df)
    
    # if not candidate_days.any():
    #     print("No suitable low volatility windows found.")
    #     return
    
    # last_candidate_date = candidate_days[candidate_days].index[-1]
    # analysis_start_date = last_candidate_date - pd.Timedelta(days=120)
    # analysis_df = df.loc[analysis_start_date:last_candidate_date]
    # print(f"Found a candidate window ending on {last_candidate_date.date()}. Analyzing...")

    # swings = find_swings_zigzag(analysis_df, threshold=0.05)
    # contractions = extract_contractions(swings)
    # vcp_result = is_valid_vcp_relaxed(contractions, need_n=2)

    # if not vcp_result:
    #     print("No valid VCP pattern found in the analysis window.")
    #     return
        
    # print(f"SUCCESS: Found a valid VCP pattern ending with pivot at {vcp_result['pivot_price']:.2f} on {vcp_result['pivot_time'].date()}.")
    # print("Contraction Depths:", [f"{p[4]*100:.1f}%" for p in vcp_result['pairs']])

    # breakout_signal = find_breakout(df, vcp_result['pivot_time'], vcp_result['pivot_price'])

    # if breakout_signal is None:
    #     print("No breakout signal found yet. The setup might still be forming.")
    # else:
    #     print(f"SUCCESS: Breakout detected on {breakout_signal.name.date()}!")
    
    # # Add backtest for single run
    # backtest_metrics = run_backtest(df, breakout_signal)
    # print("\nBacktest Results:")
    # for key, val in backtest_metrics.items():
    #     print(f"  {key}: {val}")

    # plot_vcp_analysis(df, vcp_result, breakout_signal, stock_to_test, "single_stock_test.png")


if __name__ == "__main__":
    main()

