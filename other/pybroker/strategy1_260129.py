#!/usr/bin/env python3
"""
Broad Index ETF Strategy #1: Macro Liquidity (M1-M2 Gap) + CSI300 Valuation Percentile

Production-grade implementation using AkShare public data sources.
Generates monthly target positions for CSI300 ETF based on:
- Macro liquidity trend (M1-M2 YoY gap)
- CSI300 PE(TTM) valuation percentile

Author: Quant Strategy Team
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List, Any

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def pick_col(df: pd.DataFrame, keywords: List[str], exclude: Optional[List[str]] = None) -> Optional[str]:
    """
    Fuzzy column selection based on keywords.
    
    Args:
        df: DataFrame to search columns in
        keywords: List of keywords to match (case-insensitive, any match)
        exclude: List of keywords to exclude from matches
        
    Returns:
        Best matching column name or None if no match found
    """
    exclude = exclude or []
    candidates = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        # Check if any keyword matches
        if any(kw.lower() in col_lower for kw in keywords):
            # Check exclusions
            if not any(ex.lower() in col_lower for ex in exclude):
                candidates.append(col)
    
    if not candidates:
        return None
    
    # Prefer shorter column names (more specific)
    return min(candidates, key=len)


def retry_fetch(func, max_retries: int = 3, delay: float = 2.0, **kwargs) -> Any:
    """
    Retry wrapper for data fetching functions.
    
    Args:
        func: Function to call
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds
        **kwargs: Arguments to pass to func
        
    Returns:
        Result from func
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            result = func(**kwargs)
            return result
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    logger.error(f"All {max_retries} attempts failed for {func.__name__}")
    raise last_exception


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_macro_money_supply() -> pd.DataFrame:
    """
    Fetch M1/M2 YoY data from AkShare.
    
    Returns:
        DataFrame with columns: month_end, m1_yoy, m2_yoy, gap
    """
    import akshare as ak
    
    logger.info("Fetching macro money supply data...")
    raw = retry_fetch(ak.macro_china_supply_of_money)
    
    if raw is None or raw.empty:
        raise ValueError("Empty data returned from macro_china_supply_of_money")
    
    logger.debug(f"Raw columns: {raw.columns.tolist()}")
    
    # Find date column
    date_col = pick_col(raw, ['月份', '日期', 'date', 'month', '统计时间'])
    if date_col is None:
        # Try first column if it looks like dates
        date_col = raw.columns[0]
        logger.warning(f"Using first column as date: {date_col}")
    
    # Find M1 YoY column
    m1_col = pick_col(raw, ['m1', 'M1'], exclude=['m10', 'm11', 'm12', 'm2'])
    if m1_col is None:
        m1_col = pick_col(raw, ['货币', '同比'], exclude=['m2', 'M2', '准货币'])
    
    # Find M2 YoY column  
    m2_col = pick_col(raw, ['m2', 'M2'], exclude=['m1', 'M1'])
    
    if m1_col is None or m2_col is None:
        logger.error(f"Available columns: {raw.columns.tolist()}")
        raise ValueError(f"Could not identify M1/M2 columns. M1: {m1_col}, M2: {m2_col}")
    
    logger.info(f"Using columns - Date: {date_col}, M1: {m1_col}, M2: {m2_col}")
    
    # Build clean DataFrame
    df = pd.DataFrame({
        'date_raw': raw[date_col],
        'm1_yoy': pd.to_numeric(raw[m1_col], errors='coerce'),
        'm2_yoy': pd.to_numeric(raw[m2_col], errors='coerce')
    })
    
    # Parse dates - handle various formats
    df['date'] = pd.to_datetime(df['date_raw'], errors='coerce')
    
    # If parsing failed, try common Chinese formats
    if df['date'].isna().all():
        for fmt in ['%Y年%m月', '%Y-%m', '%Y%m']:
            try:
                df['date'] = pd.to_datetime(df['date_raw'], format=fmt)
                if not df['date'].isna().all():
                    break
            except:
                continue
    
    df = df.dropna(subset=['date', 'm1_yoy', 'm2_yoy'])
    
    if df.empty:
        raise ValueError("No valid data after parsing macro money supply")
    
    # Convert to month end for consistent alignment
    df['month_end'] = df['date'] + pd.offsets.MonthEnd(0)
    
    # Compute gap
    df['gap'] = df['m1_yoy'] - df['m2_yoy']
    
    # Sort and deduplicate
    df = df.sort_values('month_end').drop_duplicates(subset=['month_end'], keep='last')
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} months of macro data from {df['month_end'].min()} to {df['month_end'].max()}")

    print(raw[[date_col, m1_col, m2_col]].tail(5))
    
    return df[['month_end', 'm1_yoy', 'm2_yoy', 'gap']]


def fetch_csi300_pe(index_name: str = "沪深300") -> pd.DataFrame:
    """
    Fetch CSI300 PE(TTM) data from AkShare.
    
    Args:
        index_name: Index name in Chinese
        
    Returns:
        DataFrame with columns: date, pe_ttm
    """
    import akshare as ak
    
    logger.info(f"Fetching PE data for {index_name}...")
    raw = retry_fetch(ak.stock_index_pe_lg, symbol=index_name)
    
    if raw is None or raw.empty:
        raise ValueError(f"Empty data returned from stock_index_pe_lg for {index_name}")
    
    logger.debug(f"Raw PE columns: {raw.columns.tolist()}")
    
    # Find date column
    date_col = pick_col(raw, ['日期', 'date', 'time'])
    if date_col is None:
        date_col = raw.columns[0]
        logger.warning(f"Using first column as date: {date_col}")
    
    # Find PE TTM column
    pe_col = pick_col(raw, ['滚动市盈率', 'pe', 'ttm', '市盈率'])
    if pe_col is None:
        # Try columns with numeric values
        for col in raw.columns:
            if col != date_col:
                if pd.to_numeric(raw[col], errors='coerce').notna().sum() > len(raw) * 0.5:
                    pe_col = col
                    break
    
    if pe_col is None:
        logger.error(f"Available columns: {raw.columns.tolist()}")
        raise ValueError("Could not identify PE column")
    
    logger.info(f"Using columns - Date: {date_col}, PE: {pe_col}")
    
    df = pd.DataFrame({
        'date': pd.to_datetime(raw[date_col], errors='coerce'),
        'pe_ttm': pd.to_numeric(raw[pe_col], errors='coerce')
    })
    
    df = df.dropna(subset=['date', 'pe_ttm'])
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} days of PE data from {df['date'].min()} to {df['date'].max()}")
    
    return df[['date', 'pe_ttm']]


# =============================================================================
# SIGNAL COMPUTATION
# =============================================================================

def compute_gap_ma3(macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 3-month rolling mean of M1-M2 gap.
    
    Args:
        macro_df: DataFrame with 'month_end' and 'gap' columns
        
    Returns:
        DataFrame with additional 'gap_ma3' column
    """
    df = macro_df.copy()
    df = df.sort_values('month_end').reset_index(drop=True)
    df['gap_ma3'] = df['gap'].rolling(window=3, min_periods=3).mean()
    return df


def compute_dynamic_threshold(
    macro_df: pd.DataFrame, 
    asof_month: pd.Timestamp,
    window_months: int = 60,
    quantile: float = 0.20
) -> float:
    """
    Compute dynamic threshold as percentile of recent gap_ma3 values.
    
    Args:
        macro_df: DataFrame with 'month_end' and 'gap_ma3' columns
        asof_month: Reference month for lookback
        window_months: Number of months to look back
        quantile: Percentile threshold (e.g., 0.20 for 20th percentile)
        
    Returns:
        Dynamic threshold value
    """
    # Get data up to and including asof_month
    mask = macro_df['month_end'] <= asof_month
    recent = macro_df.loc[mask, 'gap_ma3'].dropna().tail(window_months)
    
    if len(recent) < 12:  # Require at least 1 year of data
        logger.warning(f"Only {len(recent)} months available for threshold, using -5 as fallback")
        return -5.0
    
    return float(np.percentile(recent, quantile * 100))


def compute_trend_signal(
    current_gap_ma3: float,
    prev_gap_ma3: float,
    dyn_threshold: float
) -> int:
    """
    Compute trend signal based on liquidity conditions.
    
    Args:
        current_gap_ma3: Current month's 3-month MA of gap
        prev_gap_ma3: Previous month's 3-month MA of gap
        dyn_threshold: Dynamic threshold for liquidity_positive
        
    Returns:
        Trend signal: +1 (bullish), 0 (neutral), -1 (bearish)
    """
    liquidity_improving = current_gap_ma3 > prev_gap_ma3
    liquidity_positive = current_gap_ma3 > dyn_threshold
    
    if liquidity_improving and liquidity_positive:
        return 1
    elif not liquidity_improving:
        return -1
    else:
        return 0


def pe_percentile_rolling(
    pe_df: pd.DataFrame,
    asof_date: pd.Timestamp,
    window_years: int = 10
) -> float:
    """
    Compute rolling PE percentile with no lookahead bias.
    
    Args:
        pe_df: DataFrame with 'date' and 'pe_ttm' columns
        asof_date: Reference date (inclusive)
        window_years: Lookback window in years
        
    Returns:
        PE percentile [0, 1]
    """
    # Get data up to and including asof_date
    mask = pe_df['date'] <= asof_date
    available = pe_df.loc[mask].copy()
    
    if available.empty:
        raise ValueError(f"No PE data available on or before {asof_date}")
    
    # Get the PE value on asof_date (or closest before)
    current_pe = available['pe_ttm'].iloc[-1]
    current_date = available['date'].iloc[-1]
    
    # Get historical window
    window_start = current_date - pd.DateOffset(years=window_years)
    hist_mask = (available['date'] >= window_start) & (available['date'] <= current_date)
    hist_pe = available.loc[hist_mask, 'pe_ttm']
    
    if len(hist_pe) < 252:  # Require at least ~1 year of data
        logger.warning(f"Only {len(hist_pe)} PE observations in window, percentile may be unreliable")
    
    # Compute percentile: proportion of historical values <= current
    percentile = (hist_pe <= current_pe).mean()
    
    return float(percentile)


def valuation_weight(pe_pct: float) -> float:
    """
    Compute valuation weight from PE percentile.
    
    Continuous mapping:
    - pe_pct <= 0.20 → 1.2 (cheap, overweight)
    - pe_pct >= 0.80 → 0.0 (expensive, no exposure)
    - 0.20 < pe_pct < 0.80 → linear interpolation
    
    Args:
        pe_pct: PE percentile [0, 1]
        
    Returns:
        Valuation weight [0, 1.2]
    """
    if pe_pct <= 0.20:
        return 1.2
    elif pe_pct >= 0.80:
        return 0.0
    else:
        # Linear interpolation: w = 1.2 * (0.8 - pe_pct) / 0.6
        return 1.2 * (0.80 - pe_pct) / 0.60


def compute_target_position(
    trend_signal: int,
    val_weight: float,
    base: float = 0.50
) -> float:
    """
    Compute target position from signals.
    
    Args:
        trend_signal: Trend signal {-1, 0, +1}
        val_weight: Valuation weight [0, 1.2]
        base: Base position
        
    Returns:
        Target position clipped to [0, 1]
    """
    raw_target = (base + 0.30 * trend_signal) * val_weight
    return float(np.clip(raw_target, 0.0, 1.0))


# =============================================================================
# STATE & I/O
# =============================================================================

def load_state(state_path: Path) -> dict:
    """Load state from JSON file."""
    if state_path.exists():
        with open(state_path, 'r') as f:
            state = json.load(f)
            logger.info(f"Loaded state: {state}")
            return state
    return {}


def save_state(state_path: Path, state: dict) -> None:
    """Save state to JSON file."""
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved state: {state}")


def append_signal(
    signals_path: Path,
    row: dict
) -> None:
    """Append signal row to CSV file."""
    df = pd.DataFrame([row])
    
    write_header = not signals_path.exists()
    df.to_csv(signals_path, mode='a', header=write_header, index=False)
    logger.info(f"Appended signal to {signals_path}")


def emit_order(
    etf_symbol: str,
    target_position: float,
    macro_month: str
) -> None:
    """
    Stub for broker API integration.
    
    In production, this would:
    1. Calculate current holdings
    2. Compute required trade size
    3. Submit order via broker API
    
    Args:
        etf_symbol: ETF ticker symbol
        target_position: Target position [0, 1]
        macro_month: Reference macro month
    """
    logger.info("=" * 60)
    logger.info("ORDER EMISSION (STUB)")
    logger.info(f"  ETF Symbol: {etf_symbol}")
    logger.info(f"  Target Position: {target_position:.2%}")
    logger.info(f"  Macro Month: {macro_month}")
    logger.info("=" * 60)
    # TODO: Implement actual broker API integration
    # Example:
    # from broker_api import BrokerClient
    # client = BrokerClient()
    # current_nav = client.get_portfolio_nav()
    # target_value = current_nav * target_position
    # current_holdings = client.get_position(etf_symbol)
    # trade_value = target_value - current_holdings
    # if abs(trade_value) > MIN_TRADE_THRESHOLD:
    #     client.submit_order(etf_symbol, trade_value)


# =============================================================================
# MAIN STRATEGY
# =============================================================================

def run_strategy(
    etf_symbol: str = "510300",
    index_name: str = "沪深300",
    band: float = 0.05,
    dyn_window_months: int = 60,
    dyn_quantile: float = 0.20,
    pe_window_years: int = 10,
    outdir: Path = Path(".")
) -> None:
    """
    Main strategy execution.
    
    Args:
        etf_symbol: ETF ticker symbol
        index_name: Index name for PE data
        band: Trade band for debouncing
        dyn_window_months: Window for dynamic threshold
        dyn_quantile: Quantile for dynamic threshold
        pe_window_years: Window for PE percentile
        outdir: Output directory for files
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    state_path = outdir / "state.json"
    signals_path = outdir / "signals.csv"
    
    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Strategy run started at {run_time}")
    logger.info(f"Parameters: ETF={etf_symbol}, Index={index_name}, Band={band}")
    
    # Load state
    state = load_state(state_path)
    last_macro_month = state.get('last_macro_month')
    last_target = state.get('last_target')
    
    # Fetch data
    try:
        macro_df = fetch_macro_money_supply()
        pe_df = fetch_csi300_pe(index_name)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise
    
    # Compute gap MA3
    macro_df = compute_gap_ma3(macro_df)
    
    # Get latest complete macro month
    valid_macro = macro_df.dropna(subset=['gap_ma3'])
    if valid_macro.empty:
        logger.error("No valid gap_ma3 data available")
        return
    
    latest_month_end = valid_macro['month_end'].max()
    latest_macro_month = latest_month_end.strftime("%Y-%m")
    
    logger.info(f"Latest macro month: {latest_macro_month}, Last processed: {last_macro_month}")
    
    # Idempotency check
    if latest_macro_month == last_macro_month:
        logger.info("Macro data unchanged. SKIPPING run.")
        return
    
    # Get current and previous gap_ma3
    latest_row = valid_macro[valid_macro['month_end'] == latest_month_end].iloc[0]
    current_gap_ma3 = latest_row['gap_ma3']
    
    # Get previous month's gap_ma3
    prev_months = valid_macro[valid_macro['month_end'] < latest_month_end]
    if prev_months.empty:
        logger.error("No previous month data for trend comparison")
        return
    prev_gap_ma3 = prev_months['gap_ma3'].iloc[-1]
    
    # Compute dynamic threshold
    dyn_thr = compute_dynamic_threshold(
        macro_df, latest_month_end, dyn_window_months, dyn_quantile
    )
    logger.info(f"Gap MA3: current={current_gap_ma3:.2f}, prev={prev_gap_ma3:.2f}, dyn_thr={dyn_thr:.2f}")
    
    # Compute trend signal
    trend_signal = compute_trend_signal(current_gap_ma3, prev_gap_ma3, dyn_thr)
    logger.info(f"Trend signal: {trend_signal}")
    
    # Get PE percentile as of month end
    # Find last PE date <= month_end
    pe_mask = pe_df['date'] <= latest_month_end
    if not pe_mask.any():
        logger.error(f"No PE data available on or before {latest_month_end}")
        return
    
    pe_asof_date = pe_df.loc[pe_mask, 'date'].max()
    pe_pct = pe_percentile_rolling(pe_df, pe_asof_date, pe_window_years)
    logger.info(f"PE as-of: {pe_asof_date.strftime('%Y-%m-%d')}, percentile: {pe_pct:.2%}")
    
    # Compute valuation weight
    val_weight = valuation_weight(pe_pct)
    logger.info(f"Valuation weight: {val_weight:.2f}")
    
    # Compute target position
    target = compute_target_position(trend_signal, val_weight)
    logger.info(f"Target position: {target:.2%}")
    
    # Trade band check
    should_trade = True
    if last_target is not None:
        position_change = abs(target - last_target)
        if position_change < band:
            logger.info(f"Position change {position_change:.2%} < band {band:.2%}. No trade needed.")
            should_trade = False
    
    # Record signal
    signal_row = {
        'run_time': run_time,
        'macro_month': latest_macro_month,
        'gap_ma3': round(current_gap_ma3, 4),
        'prev_gap_ma3': round(prev_gap_ma3, 4),
        'dyn_thr_p20': round(dyn_thr, 4),
        'trend_signal': trend_signal,
        'pe_asof': pe_asof_date.strftime('%Y-%m-%d'),
        'pe_pct_10y': round(pe_pct, 4),
        'valuation_weight': round(val_weight, 4),
        'target_position': round(target, 4),
        'traded': should_trade
    }
    append_signal(signals_path, signal_row)
    
    # Emit order if needed
    if should_trade:
        emit_order(etf_symbol, target, latest_macro_month)
    
    # Update state
    new_state = {
        'last_macro_month': latest_macro_month,
        'last_target': target,
        'last_run_time': run_time
    }
    save_state(state_path, new_state)
    
    logger.info("Strategy run completed successfully.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Broad Index ETF Strategy #1: Macro Liquidity + CSI300 Valuation"
    )
    parser.add_argument(
        '--etf', type=str, default='510300',
        help='ETF symbol (default: 510300)'
    )
    parser.add_argument(
        '--index', type=str, default='沪深300',
        help='Index name for PE data (default: 沪深300)'
    )
    parser.add_argument(
        '--band', type=float, default=0.05,
        help='Trade band for debouncing (default: 0.05)'
    )
    parser.add_argument(
        '--dyn_window_months', type=int, default=60,
        help='Window months for dynamic threshold (default: 60)'
    )
    parser.add_argument(
        '--dyn_quantile', type=float, default=0.20,
        help='Quantile for dynamic threshold (default: 0.20)'
    )
    parser.add_argument(
        '--pe_window_years', type=int, default=10,
        help='PE percentile lookback window in years (default: 10)'
    )
    parser.add_argument(
        '--outdir', type=str, default='.',
        help='Output directory for signals.csv and state.json (default: .)'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        run_strategy(
            etf_symbol=args.etf,
            index_name=args.index,
            band=args.band,
            dyn_window_months=args.dyn_window_months,
            dyn_quantile=args.dyn_quantile,
            pe_window_years=args.pe_window_years,
            outdir=Path(args.outdir)
        )
    except Exception as e:
        logger.exception(f"Strategy execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
