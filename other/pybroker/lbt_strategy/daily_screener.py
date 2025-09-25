import os
import time
import random
import pandas as pd
from datetime import datetime, timedelta

# Import our project modules
import config
import data_fetcher
import indicator_calculator
import strategy
import email_sender

def format_report_html(results_df: pd.DataFrame) -> str:
    """Formats the DataFrame of results into a nice HTML report."""
    
    # Get today's date for the report title
    report_date = datetime.now().strftime("%Y-%m-%d")
    
    # Convert DataFrame to HTML table with some basic styling
    html_table = results_df.to_html(index=False, border=0, classes="styled-table")
    
    # CSS for the table
    html_style = """
    <style>
        body { font-family: Arial, sans-serif; }
        .styled-table {
            border-collapse: collapse;
            margin: 25px 0;
            font-size: 0.9em;
            min-width: 400px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.15);
        }
        .styled-table thead tr {
            background-color: #009879;
            color: #ffffff;
            text-align: left;
        }
        .styled-table th, .styled-table td {
            padding: 12px 15px;
        }
        .styled-table tbody tr {
            border-bottom: 1px solid #dddddd;
        }
        .styled-table tbody tr:nth-of-type(even) {
            background-color: #f3f3f3;
        }
        .styled-table tbody tr:last-of-type {
            border-bottom: 2px solid #009879;
        }
    </style>
    """
    
    # Combine everything into a full HTML document
    full_html = f"""
    <html>
    <head>{html_style}</head>
    <body>
        <h2>Daily LBT Stock Screener Report - {report_date}</h2>
        <p>The following stocks have triggered a buy signal (Score >= {config.SCORE_THRESHOLD}) within the last {config.SCREENER_LOOKBACK_DAYS} trading days.</p>
        {html_table}
    </body>
    </html>
    """
    return full_html

def run_daily_screener():
    """The main function to screen markets, find signals, and send a report."""
    
    start_time = time.time()
    print("--- Starting Daily LBT Screener ---")
    
    # Set the date range for fetching data (we need about a year for 252-day indicators)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    all_positive_hits = []

    # Loop through each market defined in the config
    for market in config.SCREENER_UNIVERSE:
        print(f"\n--- Scanning Market: {market} ---")
        stock_list = data_fetcher.get_stock_list(market)
        
        # Use a small slice for testing, remove slice for full run
        # stock_list = stock_list[:30] 
        
        for i, ticker in enumerate(stock_list):
            print(f"Processing {ticker} ({i+1}/{len(stock_list)})...", end="")
            
            # Fetch data for the last year
            df = data_fetcher.fetch_stock_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            print(len(df))
            if df is not None and len(df) > 252:
                # Calculate indicators and LBT scores
                df_indicators = indicator_calculator.calculate_all_indicators(df, config)
                df_scores = strategy.calculate_lbt_score(df_indicators, config)
                
                # Check the last N days for a signal
                recent_scores = df_scores.tail(config.SCREENER_LOOKBACK_DAYS)
                triggered_signals = recent_scores[recent_scores['composite_score'] >= config.SCORE_THRESHOLD]
                
                if not triggered_signals.empty:
                    # Get the most recent signal
                    latest_signal = triggered_signals.iloc[-1]
                    signal_date = latest_signal.name.strftime('%Y-%m-%d')
                    signal_price = df_indicators.loc[latest_signal.name, 'close']
                    
                    hit_details = {
                        "Market": market,
                        "Ticker": ticker,
                        "Signal Date": signal_date,
                        "Signal Price": f"{signal_price:.2f}",
                        "Score": f"{latest_signal['composite_score']:.2f}"
                    }
                    all_positive_hits.append(hit_details)
                    print(f" -> SIGNAL FOUND on {signal_date}!")
                else:
                    print(" No recent signal.")
            else:
                print(" Not enough data or fetch failed.")

            # Sleep to avoid overwhelming the API
            sleep_time = random.uniform(*config.SLEEP_INTERVAL_SECONDS)
            time.sleep(sleep_time)

    # --- Process and send the results ---
    if not all_positive_hits:
        print("\n--- Screener Finished: No new signals found today. ---")
        email_sender.send_email_notification(
            subject="Daily Stock Report: No Signals Found",
            html_content="<p>The LBT screener ran successfully, but no stocks met the criteria today.</p>"
        )
    else:
        results_df = pd.DataFrame(all_positive_hits)
        
        # Save results to a local file
        results_dir = "screener_results"
        os.makedirs(results_dir, exist_ok=True)
        filename = f"lbt_signals_{datetime.now().strftime('%Y-%m-%d')}.csv"
        results_df.to_csv(os.path.join(results_dir, filename), index=False)
        print(f"\n--- Screener Finished: Found {len(results_df)} signals. Results saved to '{filename}' ---")

        # Format and send the email report
        html_report = format_report_html(results_df)
        email_sender.send_email_notification(
            subject=f"Daily Stock Report: {len(results_df)} Signals Found!",
            html_content=html_report
        )

    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time) / 60:.2f} minutes.")


if __name__ == '__main__':
    run_daily_screener()
