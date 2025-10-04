# -*- coding: utf-8 -*-
"""
This script integrates three separate analysis components into a single VCP strategy:
1.  Detecting periods of low volatility (ATR/BBW shrinking).
2.  Identifying VCP patterns with a relaxed condition for lows.
3.  Pinpointing a high-volume breakout as a potential buy signal.
4.  Running a backtest to measure the performance of breakout signals.
5.  Visualizing the entire pattern on a candlestick chart.
This version runs on a schedule every workday at 17:00 Shanghai time.

v4: add RPS information when sending email
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
import smtplib
import email.utils
from email.mime.text import MIMEText

# --- NEW IMPORTS FOR SCHEDULING ---
import schedule
import pytz
from datetime import datetime

# =============================================================================
# SECTION 0: DATA FETCHING & UTILITIES (REVISED FOR CACHING)
# =============================================================================

def get_stock_data(stock_code: str, period: str = "daily", start_date: str = "20190101", end_date: str = "20251231") -> Optional[pd.DataFrame]:
    """
    Fetches historical stock data using Akshare with a local caching mechanism.
    - Checks for a local CSV file first.
    - If found, loads it and fetches only the new data since the last entry.
    - If not found, fetches the full history and saves it locally.
    """
    data_dir = "stock_data"
    os.makedirs(data_dir, exist_ok=True)
    file_path = os.path.join(data_dir, f"{stock_code}.csv")
    
    # --- Check for local cache ---
    if os.path.exists(file_path):
        try:
            local_df = pd.read_csv(file_path, index_col='date', parse_dates=True)
            last_date_in_file = local_df.index[-1]
            
            # If data is not up-to-date, fetch only the missing part
            if last_date_in_file.date() < pd.Timestamp.now().date():
                print(f"Updating local data for {stock_code} from {last_date_in_file.date()}...")
                update_start_date = (last_date_in_file + pd.Timedelta(days=1)).strftime('%Y%m%d')
                
                new_df = _fetch_from_akshare(stock_code, period, update_start_date, end_date)
                if new_df is not None and not new_df.empty:
                    combined_df = pd.concat([local_df, new_df])
                    combined_df = combined_df[~combined_df.index.duplicated(keep='last')] # Remove potential overlaps
                    combined_df.to_csv(file_path)
                    return combined_df.sort_index()
            
            return local_df.sort_index()
        except Exception as e:
            print(f"Error reading or updating local file for {stock_code}: {e}. Refetching all.")

    # --- If no local data, fetch all and save ---
    print(f"No local data for {stock_code}. Fetching full history from {start_date}...")
    full_df = _fetch_from_akshare(stock_code, period, start_date, end_date)
    if full_df is not None and not full_df.empty:
        full_df.to_csv(file_path)
    return full_df


def _fetch_from_akshare(stock_code: str, period: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """Internal helper to fetch data and standardize format."""
    stock_df = pd.DataFrame()
    try:
        stock_df = ak.stock_zh_a_hist(symbol=stock_code, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
    except Exception:
        pass

    if stock_df.empty:
        try:
            stock_df = ak.stock_hk_hist(symbol=stock_code, period=period, start_date=start_date, end_date=end_date, adjust="qfq")
        except Exception:
            return None

    if stock_df.empty: return None
        
    stock_df.rename(columns={'日期': 'date', '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close', '成交量': 'volume'}, inplace=True)
    
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in stock_df.columns for col in required_cols): return None
        
    stock_df = stock_df[required_cols]
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.set_index('date', inplace=True)
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        stock_df[col] = pd.to_numeric(stock_df[col], errors='coerce')
        
    stock_df.dropna(inplace=True)
    return stock_df.sort_index()


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples daily data to weekly, ending on Friday."""
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    return df.resample('W-FRI').agg(agg).dropna(how='any')


# =============================================================================
# SECTION 1: STAGE 1 - DETECT LOW VOLATILITY EXECUTION WINDOW
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
    d = df_daily.copy()
    d['ATR20_d'] = ta.atr(high=d['high'], low=d['low'], close=d['close'], length=20)
    d['BBW20_d'] = ta.bbands(close=d['close'], length=20, std=2.0).iloc[:, 3]
    d['rank_ATR_d'] = percentile_rank_last(d['ATR20_d'], window=252)
    d['rank_BBW_d'] = percentile_rank_last(d['BBW20_d'], window=252)
    d['daily_low_vol'] = (d[['rank_ATR_d', 'rank_BBW_d']].min(axis=1) <= 0.20)

    w = to_weekly(df_daily)
    w['ATR20_w'] = ta.atr(high=w['high'], low=w['low'], close=w['close'], length=20)
    w['BBW20_w'] = ta.bbands(close=w['close'], length=20, std=2.0).iloc[:, 3]
    w['rank_ATR_w'] = percentile_rank_last(w['ATR20_w'], window=52)
    w['rank_BBW_w'] = percentile_rank_last(w['BBW20_w'], window=52)
    w['weekly_low_vol'] = (w[['rank_ATR_w', 'rank_BBW_w']].min(axis=1) <= 0.35)

    is_window_open = pd.Series(False, index=df_daily.index)
    for dt, is_low_vol_week in w[w['weekly_low_vol']].iterrows():
        start_date = dt + pd.Timedelta(days=1)
        end_date = start_date + pd.Timedelta(days=exec_window_days)
        is_window_open.loc[start_date:end_date] = True

    candidates = is_window_open & d['daily_low_vol'].reindex(df_daily.index).fillna(False)
    return candidates

# =============================================================================
# SECTION 2: STAGE 2 - VCP PATTERN DETECTION
# =============================================================================
ContractionPairs = List[Tuple[pd.Timestamp, float, pd.Timestamp, float, float]]

def find_swings_zigzag(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
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
        elif direction == 1:
            if hi > max_price: max_price, max_i = hi, i
            elif (max_price - lo) / max_price >= threshold and max_i < i:
                swing_high[max_i] = max_price
                direction, min_price, min_i = -1, lo, i
        elif direction == -1:
            if lo < min_price: min_price, min_i = lo, i
            elif (hi - min_price) / min_price >= threshold and min_i < i:
                swing_low[min_i] = min_price
                direction, max_price, max_i = 1, hi, i
                
    return pd.DataFrame({"swing_high": swing_high, "swing_low": swing_low}, index=df.index)

def extract_contractions(swings: pd.DataFrame, min_drop: float = 0.03, max_drop: float = 0.40, min_bars: int = 5) -> ContractionPairs:
    pairs = []
    sh = swings['swing_high'].dropna()
    sl = swings['swing_low'].dropna()
    
    for t_hi, v_hi in sh.items():
        possible_lows = sl[sl.index > t_hi]
        if not possible_lows.empty:
            t_lo, v_lo = possible_lows.index[0], possible_lows.iloc[0]
            
            prev_highs = sh[sh.index < t_lo]
            if not prev_highs.empty and prev_highs.index[-1] != t_hi:
                continue
                
            if (t_lo - t_hi).days < min_bars: continue
            
            drop = (v_hi - v_lo) / v_hi
            if min_drop <= drop <= max_drop:
                pairs.append((t_hi, v_hi, t_lo, v_lo, drop))
                
    return pairs

def is_valid_vcp_relaxed(pairs: ContractionPairs, need_n: int = 2) -> Optional[Dict]:
    if len(pairs) < need_n: return None

    sub_pairs = pairs[-need_n:]
    drops = [p[4] for p in sub_pairs]
    lows = [p[3] for p in sub_pairs]
    
    for i in range(len(drops) - 1):
        if drops[i] <= drops[i+1]:
            return None

    first_low = lows[0]
    for i in range(1, len(lows)):
        if lows[i] <= first_low:
            return None

    pivot_time = sub_pairs[-1][0]
    pivot_price = sub_pairs[-1][1]

    return {"valid": True, "pairs": sub_pairs, "pivot_time": pivot_time, "pivot_price": pivot_price}

# =============================================================================
# SECTION 3: STAGE 3 - BREAKOUT DETECTION
# =============================================================================

def find_breakout(df: pd.DataFrame, pivot_time: pd.Timestamp, pivot_price: float, look_ahead: int = 20) -> Optional[pd.Series]:
    scan_start = pivot_time + pd.Timedelta(days=1)
    scan_end = scan_start + pd.Timedelta(days=look_ahead)
    scan_df = df.loc[scan_start:scan_end]

    if scan_df.empty: return None

    price_break = scan_df['high'] > pivot_price
    vol_sma50 = df['volume'].rolling(50, min_periods=20).mean()
    volume_break = scan_df['volume'] > 1.8 * vol_sma50.reindex(scan_df.index)
    
    breakouts = scan_df[price_break & volume_break]
    
    if not breakouts.empty:
        return breakouts.iloc[0]
    return None

# =============================================================================
# SECTION 4: VISUALIZATION (FINAL REVISION)
# =============================================================================

def plot_vcp_analysis(df: pd.DataFrame, vcp_result: Dict, breakout_signal: Optional[pd.Series], execution_windows: pd.Series, stock_code: str, save_path: str):
    """
    Plots the candlestick chart with VCP contractions, pivot, breakout point,
    and highlights the low-volatility execution windows.
    (REVISED to use the correct 'fill_between' argument for highlighting)
    """
    vcp_pairs = vcp_result['pairs']
    pivot_price = vcp_result['pivot_price']
    
    start_date = vcp_pairs[0][0] - pd.Timedelta(days=30)
    end_date = breakout_signal.name + pd.Timedelta(days=15) if breakout_signal is not None else df.index[-1]
    plot_df = df.loc[start_date:end_date]
    
    if plot_df.empty:
        print(f"Warning: Plotting range for {stock_code} is empty. Skipping chart.")
        return

    # Convert timestamps to string format for 'alines' to prevent plotting errors.
    contraction_lines = [
        [(p[0].strftime('%Y-%m-%d'), p[1]), (p[2].strftime('%Y-%m-%d'), p[3])] 
        for p in vcp_pairs
    ]
        
    pivot_start_str = vcp_pairs[-1][0].strftime('%Y-%m-%d')
    pivot_end_str = (breakout_signal.name if breakout_signal is not None else plot_df.index[-1]).strftime('%Y-%m-%d')
    pivot_line = [(pivot_start_str, pivot_price), (pivot_end_str, pivot_price)]
    
    all_lines = contraction_lines + [pivot_line]
    line_colors = ['blue'] * len(contraction_lines) + ['red']
    
    add_plots = []
    
    # --- Marker for breakout point ---
    if breakout_signal is not None:
        buy_marker_df = pd.DataFrame(index=plot_df.index)
        buy_marker_df['signal'] = np.nan
        buy_marker_df.loc[breakout_signal.name, 'signal'] = breakout_signal['low'] * 0.98
        buy_plot = mpf.make_addplot(buy_marker_df['signal'], type='scatter', marker='^', color='green', markersize=200, panel=0)
        add_plots.append(buy_plot)
    
    # --- Prepare data for fill_between ---
    plot_min = plot_df['low'].min()
    plot_max = plot_df['high'].max()
    windows_in_plot = execution_windows.reindex(plot_df.index).fillna(False)
    
    # Create the y1 and y2 series for the fill
    fill_y1 = pd.Series(np.where(windows_in_plot, plot_min, np.nan), index=plot_df.index)
    fill_y2 = pd.Series(np.where(windows_in_plot, plot_max, np.nan), index=plot_df.index)

    # The fill_between argument takes a dictionary
    fill_dict = dict(y1=fill_y1.values, y2=fill_y2.values, alpha=0.15, color='yellow')

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
        fill_between=fill_dict, # Use the correct argument here
        returnfig=True,
        panel_ratios=(3, 1)
    )
    
    # --- Add text labels for contraction lows ---
    ax = axlist[0] # Main price panel axis
    for i, p in enumerate(vcp_pairs):
         # Add a check to ensure the date is within the plot's x-axis limits
         if p[2] in plot_df.index:
            ax.text(p[2], p[3], f' L{i+1}', verticalalignment='top', fontsize=9, color='black')
    
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    
# =============================================================================
# SECTION 5: BACKTESTING ENGINE
# =============================================================================
def run_backtest(df: pd.DataFrame, breakout_signal: pd.Series) -> Dict:
    backtest_results = {}
    breakout_date = breakout_signal.name
    entry_price = breakout_signal['close']
    
    holding_periods = {'20d_ret': 20, '60d_ret': 60}
    for name, days in holding_periods.items():
        exit_date_loc = df.index.get_loc(breakout_date) + days
        if exit_date_loc < len(df.index):
            exit_price = df.iloc[exit_date_loc]['close']
            ret = (exit_price - entry_price) / entry_price
            backtest_results[name] = f"{ret:.2%}"
        else:
            backtest_results[name] = "N/A"
            
    stop_loss_pct, take_profit_pct, max_hold_days = -0.08, 0.25, 120
    stop_loss_price = entry_price * (1 + stop_loss_pct)
    take_profit_price = entry_price * (1 + take_profit_pct)
    trade_df = df.loc[df.index > breakout_date].iloc[:max_hold_days]
    
    exit_reason = "Max Hold"
    exit_price = trade_df.iloc[-1]['close'] if not trade_df.empty else entry_price
    days_held = len(trade_df)

    for i, row in trade_df.iterrows():
        if row['low'] <= stop_loss_price:
            exit_reason, exit_price, days_held = "Stop-Loss", stop_loss_price, (i - breakout_date).days
            break
        if row['high'] >= take_profit_price:
            exit_reason, exit_price, days_held = "Take-Profit", take_profit_price, (i - breakout_date).days
            break
            
    strategy_ret = (exit_price - entry_price) / entry_price
    backtest_results['strat_ret'] = f"{strategy_ret:.2%}"
    backtest_results['strat_exit_reason'] = exit_reason
    backtest_results['strat_days_held'] = days_held
    
    return backtest_results

# =============================================================================
# SECTION 6: EMAIL NOTIFICATION (REVISED FOR SUMMARY REPORT)
# =============================================================================
def format_summary_email_html(patterns_list: List[Dict], breakouts_list: List[Dict]) -> str:
    """Creates a single, well-formatted HTML email body for all findings in a cycle."""
    
    alert_time = pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S %Z')

    html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        h2 {{ color: #1e90ff; border-bottom: 2px solid #1e90ff; padding-bottom: 5px;}}
        h3 {{ color: #4682b4; }}
        p {{ font-size: 0.9em; color: #666; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1.5em 0; }}
        th, td {{ border: 1px solid #dddddd; text-align: left; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
        .no-results {{font-style: italic; color: #888;}}
    </style>
    </head>
    <body>
    <h2>VCP Stock Screener Summary</h2>
    <p>Report generated on: {alert_time}</p>
    """

    # --- Section for VCP Patterns Found ---
    html += "<h3>New VCP Patterns Detected (Awaiting Breakout)</h3>"
    if patterns_list:
        patterns_df = pd.DataFrame(patterns_list)
        html += patterns_df.to_html(index=False)
    else:
        html += "<p class='no-results'>No new VCP patterns were found in this cycle.</p>"
        
    # --- Section for Confirmed Breakouts ---
    html += "<h3>New VCP Breakouts Confirmed</h3>"
    if breakouts_list:
        breakouts_df = pd.DataFrame(breakouts_list)
        # Reorder columns for better readability
        cols_order = [
            'Market', 'Stock Code', 'Stock Name', 'Breakout Date', 'Breakout Close', 
            'Pivot Time', 'Pivot Price', 'strat_ret', 'strat_exit_reason', 'strat_days_held',
            '20d_ret', '60d_ret', 'Contraction Depths'
        ]
        # Make sure all columns exist before reordering
        final_cols = [col for col in cols_order if col in breakouts_df.columns]
        breakouts_df = breakouts_df[final_cols]
        html += breakouts_df.to_html(index=False)
    else:
        html += "<p class='no-results'>No new recent breakouts were found in this cycle.</p>"

    html += "</body></html>"
    return html

def send_email_notification(subject: str, html_content: str):
    """Sends an email with the given subject and HTML content."""
    
    message = MIMEText(html_content, 'html', 'utf-8')
    message['To'] = email.utils.formataddr(('sun', 'sunweihao97@gmail.com'))
    message['From'] = email.utils.formataddr(('weihao', '2309598788@qq.com'))
    message['Subject'] = subject
    
    server = None
    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', 465)
        server.login('2309598788@qq.com', 'aoyqtzjhzmxaeafg')
        server.sendmail('2309598788@qq.com', ['sunweihao97@gmail.com'], msg=message.as_string())
        print(f"Successfully sent summary email notification: '{subject}'")
    except Exception as e:
        print(f"Failed to send summary email. Error: {e}")
    finally:
        if server:
            server.quit()

# =============================================================================
# SECTION 7: MARKET SCREENING WORKFLOW (REVISED FOR BATCH REPORTING)
# =============================================================================
def run_market_screen(stock_list_df: pd.DataFrame, market_name: str, notified_patterns: set, notified_breakouts: set, start_date: str, end_date: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Screens stocks, collects VCP patterns and recent breakouts, and returns them in lists.
    """
    total_stocks = len(stock_list_df)
    print(f"Starting to screen {total_stocks} stocks in the {market_name} market...")
    
    chart_dir = "vcp_charts"
    os.makedirs(chart_dir, exist_ok=True)
    
    found_patterns = []
    found_breakouts = []

    for i, stock_row in stock_list_df.iterrows():
        stock_code = stock_row['代码']
        stock_name = stock_row['名称']
        
        print(f"\n[{i+1}/{total_stocks} in {market_name}] Analyzing: {stock_code} ({stock_name})")
        
        sleep_time = random.uniform(0.5, 3.5)
        
        try:
            df = get_stock_data(stock_code, start_date=start_date, end_date=end_date)
            if df is None or len(df) < 200:
                time.sleep(sleep_time)
                continue

            candidate_days = find_execution_windows(df)
            if not candidate_days.any():
                time.sleep(sleep_time)
                continue
            
            last_candidate_date = candidate_days[candidate_days].index[-1]
            analysis_df = df.loc[last_candidate_date - pd.Timedelta(days=120):last_candidate_date]

            swings = find_swings_zigzag(analysis_df, threshold=0.05)
            contractions = extract_contractions(swings)
            vcp_result = is_valid_vcp_relaxed(contractions, need_n=2)
            
            if not vcp_result:
                time.sleep(sleep_time)
                continue

            # --- Stage 1 Found: VCP Pattern ---
            pattern_id = (stock_code, vcp_result['pivot_time'].date())
            if pattern_id not in notified_patterns:
                notified_patterns.add(pattern_id) # Add to set immediately to prevent re-processing

                if (pd.Timestamp.now().normalize() - vcp_result['pivot_time'].normalize()).days <= 15:
                    print(f"SUCCESS: Found RECENT VCP Pattern for {stock_code}. Adding to summary.")
                    vcp_data = {
                        "Market": market_name, "Stock Code": stock_code, "Stock Name": stock_name,
                        "Pivot Time": vcp_result['pivot_time'].date(),
                        "Pivot Price": f"{vcp_result['pivot_price']:.2f}",
                        "Contraction Depths": ", ".join([f"{p[4]*100:.1f}%" for p in vcp_result['pairs']])
                    }
                    found_patterns.append(vcp_data)
                else:
                    print(f"Found old VCP pattern for {stock_code} (pivot: {vcp_result['pivot_time'].date()}). Skipping pattern summary.")
            
            # --- Check for Breakout ---
            breakout_signal = find_breakout(df, vcp_result['pivot_time'], vcp_result['pivot_price'])
            if breakout_signal is None:
                time.sleep(sleep_time)
                continue

            # --- Stage 2 Found: Breakout Confirmed ---
            if (pd.Timestamp.now().normalize() - breakout_signal.name.normalize()).days > 5:
                print(f"Found old breakout for {stock_code}. Skipping.")
                time.sleep(sleep_time)
                continue

            breakout_id = (stock_code, breakout_signal.name.date())
            if breakout_id in notified_breakouts:
                time.sleep(sleep_time)
                continue

            print(f"SUCCESS: Found RECENT Breakout for {stock_code}. Adding to summary.")
            notified_breakouts.add(breakout_id)
            backtest_metrics = run_backtest(df, breakout_signal)
            
            breakout_data = {
                "Market": market_name, "Stock Code": stock_code, "Stock Name": stock_name,
                "Pivot Time": vcp_result['pivot_time'].date(),
                "Pivot Price": f"{vcp_result['pivot_price']:.2f}",
                "Contraction Depths": ", ".join([f"{p[4]*100:.1f}%" for p in vcp_result['pairs']]),
                "Breakout Date": breakout_signal.name.date(),
                "Breakout Close": f"{breakout_signal['close']:.2f}",
                **backtest_metrics
            }
            found_breakouts.append(breakout_data)
            
            breakout_date_str = breakout_signal.name.strftime('%Y-%m-%d')
            safe_stock_name = "".join([c for c in stock_name if c.isalnum()]).rstrip()
            file_name = f"{stock_code}_{safe_stock_name}_{breakout_date_str}.png"
            save_path = os.path.join(chart_dir, file_name)
            
            print(f"Saving chart to {save_path}...")
            plot_vcp_analysis(df, vcp_result, breakout_signal, candidate_days, f"{stock_code} - {stock_name}", save_path)

        except Exception as e:
            print(f"An error occurred while processing {stock_code}: {e}")
        
        finally:
            print(f"Sleeping for {sleep_time:.1f} seconds...")
            time.sleep(sleep_time)
            
    return found_patterns, found_breakouts

# =============================================================================
# SECTION 8: SCHEDULING AND MAIN EXECUTION
# =============================================================================

# Sets are defined globally so they persist across scheduled runs
notified_patterns = set()
notified_breakouts = set()

def run_screening_job():
    """
    This is the main job function that the scheduler will call.
    It runs one full screening cycle for both A-Share and Hong Kong markets.
    """
    print(f"\n{'='*30} JOB STARTING {'='*30}")
    job_start_time = datetime.now(pytz.timezone('Asia/Shanghai'))
    print(f"Starting scheduled screening job at: {job_start_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    start_date="20190101"
    end_date = pd.Timestamp.now().strftime('%Y%m%d')
    
    all_new_patterns = []
    all_new_breakouts = []
    
    # --- A-Shares Screening ---
    print("\n" + "="*80)
    print(f"Starting A-Share Market Scan...")
    try:
        all_stocks_df_a = ak.stock_zh_a_spot_em()
        patterns_a, breakouts_a = run_market_screen(all_stocks_df_a, "A-Shares", notified_patterns, notified_breakouts, start_date, end_date)
        all_new_patterns.extend(patterns_a)
        all_new_breakouts.extend(breakouts_a)
    except Exception as e:
        print(f"FATAL ERROR during A-share list fetching/screening: {e}")
    
    # --- Hong Kong Shares Screening ---
    print("\n" + "="*80)
    print(f"Starting Hong Kong Market Scan...")
    try:
        all_stocks_df_hk = ak.stock_hk_spot_em()
        patterns_hk, breakouts_hk = run_market_screen(all_stocks_df_hk, "Hong Kong", notified_patterns, notified_breakouts, start_date, end_date)
        all_new_patterns.extend(patterns_hk)
        all_new_breakouts.extend(breakouts_hk)
    except Exception as e:
        print(f"FATAL ERROR during Hong Kong list fetching/screening: {e}")

    # --- Send one summary email for the entire cycle ---
    if all_new_patterns or all_new_breakouts:
        print("\n" + "="*80)
        print("New signals found. Preparing and sending summary email...")
        
        subject = f"VCP Screener Summary - {pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M')}"
        html_body = format_summary_email_html(all_new_patterns, all_new_breakouts)
        send_email_notification(subject, html_body)
    else:
        print("\n" + "="*80)
        print("No new signals found in this screening job.")
    
    print(f"\n{'='*30} JOB FINISHED {'='*30}\n")


def main():
    """
    Main function to set up and run the scheduler.
    """
    # Define the timezone for Shanghai
    shanghai_tz = pytz.timezone('Asia/Shanghai')

    # Schedule the job to run every workday at 17:00 Shanghai time
    schedule.every().monday.at("10:00", shanghai_tz).do(run_screening_job)
    schedule.every().tuesday.at("10:15", shanghai_tz).do(run_screening_job)
    schedule.every().wednesday.at("10:00", shanghai_tz).do(run_screening_job)
    schedule.every().thursday.at("10:00", shanghai_tz).do(run_screening_job)
    schedule.every().friday.at("10:00", shanghai_tz).do(run_screening_job)

    print("✅ Scheduler started successfully.")
    print("The script will run every workday (Mon-Fri) at 10:00 Shanghai time.")
    print("Keep this script running in the background.")
    
    # Run the job once immediately on startup if you want
    # print("\nRunning one initial scan right now...")
    # run_screening_job()
    
    while True:
        # Check if any scheduled jobs are pending
        schedule.run_pending()
        # Wait for a minute before checking again to conserve CPU
        time.sleep(60)

if __name__ == "__main__":
    main()