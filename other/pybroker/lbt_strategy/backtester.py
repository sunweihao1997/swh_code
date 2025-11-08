import pandas as pd
import numpy as np

def run_backtest(all_scores, all_data, config):
    """Runs a more robust, day-by-day backtest on the generated scores."""
    print("Running backtest...")
    
    # We will build the portfolio history as a list of dictionaries
    portfolio_history = []
    
    active_positions = {}  # {ticker: {'entry_price': X, 'shares': Y, 'entry_date': Z}}
    cash = config.INITIAL_CAPITAL

    # Loop through each trading day in the scores DataFrame
    for date in all_scores.index:
        
        # --- 1. Update Holdings Value & Process Exits ---
        current_holdings_value = 0
        positions_to_close = []
        for ticker, pos in active_positions.items():
            # Check if the stock has data for the current date
            if ticker in all_data and date in all_data[ticker].index:
                current_price = all_data[ticker].loc[date, 'close']
                
                # Check for stop-loss or take-profit
                if current_price <= pos['entry_price'] * (1 - config.STOP_LOSS_PCT) or \
                   current_price >= pos['entry_price'] * (1 + config.PROFIT_TAKE_PCT):
                    
                    # Sell the position
                    sale_proceeds = current_price * pos['shares'] * (1 - config.COMMISSION_BPS / 10000)
                    cash += sale_proceeds
                    
                    trade_log_entry = {
                        'ticker': ticker, 'entry_date': pos['entry_date'], 'exit_date': date,
                        'entry_price': pos['entry_price'], 'exit_price': current_price,
                        'pnl': (current_price - pos['entry_price']) * pos['shares']
                    }
                    # We will create the trade_log DataFrame at the end
                    positions_to_close.append(trade_log_entry)
                else:
                    # If not selling, its value contributes to today's holdings
                    current_holdings_value += current_price * pos['shares']
            else:
                # If stock has no data (e.g., suspension), use last known price for valuation
                current_holdings_value += pos['entry_price'] * pos['shares']

        # Remove closed positions from the active list
        for closed_trade in positions_to_close:
            del active_positions[closed_trade['ticker']]

        # --- 2. Process New Entries ---
        daily_scores = all_scores.loc[date].sort_values(ascending=False)
        buy_signals = daily_scores[daily_scores >= config.SCORE_THRESHOLD]
        
        available_slots = config.MAX_ACTIVE_POSITIONS - len(active_positions)
        if available_slots > 0:
            cash_per_position = cash / available_slots # Allocate remaining cash

            for ticker in buy_signals.index:
                if available_slots <= 0: break
                if ticker not in active_positions and ticker in all_data and date in all_data[ticker].index:
                    price = all_data[ticker].loc[date, 'open'] # Buy on the day's open
                    
                    if pd.notna(price) and price > 0:
                        shares = (cash_per_position / price)
                        cost = shares * price * (1 + config.COMMISSION_BPS / 10000)
                        
                        if cash >= cost:
                            cash -= cost
                            active_positions[ticker] = {'entry_price': price, 'shares': shares, 'entry_date': date}
                            available_slots -= 1
                            # Update holdings value with the new position for today's record
                            current_holdings_value += shares * price

        # --- 3. Record Daily Portfolio Value ---
        total_value = cash + current_holdings_value
        portfolio_history.append({
            'date': date,
            'cash': cash,
            'holdings_value': current_holdings_value,
            'total_value': total_value
        })

    # --- 4. Finalize and Return Results ---
    portfolio_df = pd.DataFrame(portfolio_history).set_index('date')
    trade_log_df = pd.DataFrame([t for t in positions_to_close if 'exit_date' in t])

    return portfolio_df, trade_log_df
