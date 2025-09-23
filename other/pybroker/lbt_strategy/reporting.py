import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

def generate_report(portfolio, trade_log, benchmark_data, config):
    """Generates a performance report and equity curve chart."""
    print("Generating report...")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    # --- Align benchmark data to portfolio dates ---
    benchmark_data = benchmark_data.reindex(portfolio.index).fillna(method='ffill')

    # --- Calculate Metrics ---
    total_return = (portfolio['total_value'].iloc[-1] / config.INITIAL_CAPITAL) - 1
    benchmark_return = (benchmark_data['close'].iloc[-1] / benchmark_data['close'].iloc[0]) - 1
    
    daily_returns = portfolio['total_value'].pct_change().dropna()
    sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() != 0 else 0
    
    cumulative = (1 + daily_returns).cumprod()
    peak = cumulative.expanding(min_periods=1).max()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()

    win_rate = 0
    if trade_log is not None and not trade_log.empty:
        win_rate = (trade_log['pnl'] > 0).sum() / len(trade_log)

    # --- Write Text Report ---
    report_path = os.path.join(results_dir, 'backtest_report.txt')
    with open(report_path, 'w') as f:
        f.write("--- Backtest Performance Report ---\n")
        f.write(f"Period: {config.START_DATE} to {config.END_DATE}\n")
        f.write("-" * 35 + "\n")
        f.write(f"Total Return: {total_return:.2%}\n")
        f.write(f"Benchmark Return: {benchmark_return:.2%}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio:.2f}\n")
        f.write(f"Max Drawdown: {max_drawdown:.2%}\n")
        f.write(f"Win Rate: {win_rate:.2%}\n")
        f.write(f"Number of Trades: {len(trade_log) if trade_log is not None else 0}\n")

    # --- Save Trade Log ---
    if trade_log is not None:
        trade_log.to_csv(os.path.join(results_dir, 'trade_log.csv'))

    # --- Plot Equity Curve ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot((portfolio['total_value'] / config.INITIAL_CAPITAL), label='LBT Strategy', color='royalblue')
    ax.plot((benchmark_data['close'] / benchmark_data['close'].iloc[0]), label='Benchmark', color='darkorange', linestyle='--')
    
    ax.set_title('Equity Curve: LBT Strategy vs. Benchmark', fontsize=16)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True)
    
    chart_path = os.path.join(results_dir, 'equity_curve.png')
    plt.savefig(chart_path)
    plt.close()
    
    print(f"Report and chart saved to '{results_dir}' directory.")
