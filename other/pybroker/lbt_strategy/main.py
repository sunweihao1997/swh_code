import pandas as pd
import config
import data_fetcher
import indicator_calculator
import strategy
import backtester
import reporting

def main():
    """Main function to run the LBT screening and backtesting framework."""
    print(f"Starting LBT backtest from {config.START_DATE} to {config.END_DATE}...")
    
    # Note: Analyzing the full stock list can be very slow.
    # We limit to the first 50 stocks for a faster demonstration.
    # Remove the '[:50]' slice to run on the full market.
    stock_list = data_fetcher.get_stock_list(config.STOCK_UNIVERSE)[:50]
    all_data = {}
    
    print("--- Phase 1: Fetching and Processing Stock Data ---")
    for ticker in stock_list:
        df = data_fetcher.fetch_stock_data(ticker, config.START_DATE, config.END_DATE)
        # Ensure enough data for indicators (e.g., 252 for 52-week high/low)
        if df is not None and len(df) > 252: 
            df_with_indicators = indicator_calculator.calculate_all_indicators(df, config)
            # Apply liquidity filter
            if df_with_indicators['turnover'].rolling(window=20).mean().iloc[-1] > config.MIN_AVG_TURNOVER:
                all_data[ticker] = df_with_indicators

    if not all_data:
        print("No stocks passed the initial data fetching and liquidity filters. Exiting.")
        return

    print(f"--- Phase 2: Calculating Strategy Scores for {len(all_data)} stocks ---")
    all_scores = {}
    for ticker, df in all_data.items():
        all_scores[ticker] = strategy.calculate_lbt_score(df, config)['composite_score']
    
    scores_df = pd.DataFrame(all_scores)
    # Forward-fill scores to handle non-trading days for the backtester index
    scores_df.ffill(inplace=True)
    
    print("--- Phase 3: Running Backtest Simulation ---")
    portfolio, trade_log = backtester.run_backtest(scores_df, all_data, config)
    
    print("--- Phase 4: Generating Final Report ---")
    benchmark_data = data_fetcher.fetch_stock_data(config.BENCHMARK_TICKER, config.START_DATE, config.END_DATE)
    
    # --- THIS IS THE FIX ---
    # Check if benchmark data was successfully fetched before generating the report.
    if benchmark_data is None:
        print(f"CRITICAL ERROR: Could not fetch benchmark data for '{config.BENCHMARK_TICKER}'.")
        print("The backtest ran, but the report cannot be generated without the benchmark.")
        print("Please check the ticker symbol and your network connection. Exiting.")
        return # Exit the main function gracefully
        
    reporting.generate_report(portfolio, trade_log, benchmark_data, config)
    print("--- Process Complete ---")

if __name__ == '__main__':
    main()
