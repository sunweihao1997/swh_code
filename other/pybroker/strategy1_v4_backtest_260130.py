#!/usr/bin/env python3
"""
Backtest Framework for Strategy #1: Macro Liquidity + CSI300 Valuation

Evaluates strategy performance across different parameter configurations
over 10-year and 20-year historical periods.

Features:
- Month-by-month simulation with no lookahead bias
- Parameter grid search
- Performance metrics: returns, Sharpe ratio, max drawdown, etc.
- Comparison against buy-and-hold benchmark
- Results export to CSV and visualization

Author: Quant Strategy Team
"""

import argparse
import itertools
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# =============================================================================
# STRATEGY FUNCTIONS (imported logic from strategy1_v2)
# =============================================================================

def pick_col(df: pd.DataFrame, keywords: List[str], exclude: Optional[List[str]] = None) -> Optional[str]:
    """Fuzzy column selection based on keywords."""
    exclude = exclude or []
    candidates = []
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw.lower() in col_lower for kw in keywords):
            if not any(ex.lower() in col_lower for ex in exclude):
                candidates.append(col)
    if not candidates:
        return None
    return min(candidates, key=len)


def retry_fetch(func, max_retries: int = 3, delay: float = 2.0, **kwargs) -> Any:
    """Retry wrapper for data fetching functions."""
    last_exception = None
    for attempt in range(max_retries):
        try:
            return func(**kwargs)
        except Exception as e:
            last_exception = e
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(delay * (attempt + 1))
    raise last_exception


def compute_gap_ma3(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Compute 3-month rolling mean of M1-M2 gap."""
    df = macro_df.copy()
    df = df.sort_values('month_end').reset_index(drop=True)
    df['gap_ma3'] = df['gap'].rolling(window=3, min_periods=3).mean()
    return df


def compute_dynamic_threshold(
    macro_df: pd.DataFrame,
    asof_month: pd.Timestamp,
    window_months: int = 60,
    quantile: float = 0.20,
    exclude_current: bool = False
) -> float:
    """
    Compute dynamic threshold as percentile of recent gap_ma3 values.
    
    Args:
        macro_df: DataFrame with 'month_end' and 'gap_ma3' columns
        asof_month: Reference month for lookback
        window_months: Number of months to look back
        quantile: Percentile threshold (e.g., 0.20 for 20th percentile)
        exclude_current: If True, exclude asof_month from the calculation
                        to avoid lookahead bias (use data up to asof_month - 1)
    
    Returns:
        Dynamic threshold value
    """
    if exclude_current:
        # Use data strictly before asof_month (up to asof_month - 1 month)
        mask = macro_df['month_end'] < asof_month
    else:
        # Use data up to and including asof_month
        mask = macro_df['month_end'] <= asof_month
    
    recent = macro_df.loc[mask, 'gap_ma3'].dropna().tail(window_months)
    if len(recent) < 12:
        return -5.0
    return float(np.percentile(recent, quantile * 100))


def compute_trend_signal(
    current_gap_ma3: float,
    prev_gap_ma3: float,
    dyn_threshold: float
) -> int:
    """Compute trend signal based on liquidity conditions."""
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
) -> Optional[float]:
    """Compute rolling PE percentile with no lookahead bias."""
    mask = pe_df['date'] <= asof_date
    available = pe_df.loc[mask].copy()
    if available.empty:
        return None
    current_pe = available['pe_ttm'].iloc[-1]
    current_date = available['date'].iloc[-1]
    window_start = current_date - pd.DateOffset(years=window_years)
    hist_mask = (available['date'] >= window_start) & (available['date'] <= current_date)
    hist_pe = available.loc[hist_mask, 'pe_ttm']
    if len(hist_pe) < 60:  # Require minimum data
        return None
    return float((hist_pe <= current_pe).mean())


def valuation_weight(
    pe_pct: float,
    cheap_threshold: float = 0.20,
    expensive_threshold: float = 0.80,
    max_weight: float = 1.2
) -> float:
    """Compute valuation weight from PE percentile with configurable thresholds."""
    if pe_pct <= cheap_threshold:
        return max_weight
    elif pe_pct >= expensive_threshold:
        return 0.0
    else:
        return max_weight * (expensive_threshold - pe_pct) / (expensive_threshold - cheap_threshold)


def compute_target_position(
    trend_signal: int,
    val_weight: float,
    base: float = 0.50,
    trend_sensitivity: float = 0.30
) -> float:
    """Compute target position from signals."""
    raw_target = (base + trend_sensitivity * trend_signal) * val_weight
    return float(np.clip(raw_target, 0.0, 1.0))


# =============================================================================
# DATA FETCHING
# =============================================================================

def fetch_macro_money_supply() -> pd.DataFrame:
    """Fetch M1/M2 YoY data from AkShare."""
    import akshare as ak
    
    logger.info("Fetching macro money supply data...")
    raw = retry_fetch(ak.macro_china_supply_of_money)
    
    if raw is None or raw.empty:
        raise ValueError("Empty data from macro_china_supply_of_money")
    
    date_col = pick_col(raw, ['月份', '日期', 'date', 'month', '统计时间'])
    if date_col is None:
        date_col = raw.columns[0]
    
    # Find M1/M2 YoY columns
    def _find_money_yoy_col(target: str) -> Optional[str]:
        yoy_markers = ['同比', '增速', '增长', 'yoy']
        candidates = []
        for col in raw.columns:
            s = str(col)
            sl = s.lower()
            if target == 'm1':
                if ('m1' not in sl) or ('m2' in sl):
                    continue
                if any(x in sl for x in ['m10', 'm11', 'm12']):
                    continue
            elif target == 'm2':
                if 'm2' not in sl:
                    continue
                if 'm1' in sl:
                    continue
            if any(m in s for m in yoy_markers) or any(m in sl for m in yoy_markers):
                candidates.append(col)
        if not candidates:
            return None
        return min(candidates, key=lambda x: len(str(x)))
    
    m1_col = _find_money_yoy_col('m1')
    m2_col = _find_money_yoy_col('m2')
    
    if m1_col is None or m2_col is None:
        raise ValueError(f"Could not identify M1/M2 YoY columns. Found M1: {m1_col}, M2: {m2_col}")
    
    logger.info(f"Using columns - Date: {date_col}, M1: {m1_col}, M2: {m2_col}")
    
    df = pd.DataFrame({
        'date_raw': raw[date_col],
        'm1_yoy': pd.to_numeric(raw[m1_col], errors='coerce'),
        'm2_yoy': pd.to_numeric(raw[m2_col], errors='coerce')
    })
    
    df['date'] = pd.to_datetime(df['date_raw'], errors='coerce')
    if df['date'].isna().all():
        for fmt in ['%Y年%m月', '%Y-%m', '%Y%m', '%Y.%m']:
            try:
                df['date'] = pd.to_datetime(df['date_raw'], format=fmt)
                if not df['date'].isna().all():
                    break
            except:
                continue
    
    df = df.dropna(subset=['date', 'm1_yoy', 'm2_yoy'])
    df['month_end'] = df['date'] + pd.offsets.MonthEnd(0)
    df['gap'] = df['m1_yoy'] - df['m2_yoy']
    df = df.sort_values('month_end').drop_duplicates(subset=['month_end'], keep='last')
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} months of macro data: {df['month_end'].min()} to {df['month_end'].max()}")
    return df[['month_end', 'm1_yoy', 'm2_yoy', 'gap']]


def fetch_csi300_pe(index_name: str = "沪深300") -> pd.DataFrame:
    """Fetch CSI300 PE(TTM) data from AkShare."""
    import akshare as ak
    
    logger.info(f"Fetching PE data for {index_name}...")
    raw = retry_fetch(ak.stock_index_pe_lg, symbol=index_name)
    
    if raw is None or raw.empty:
        raise ValueError(f"Empty data from stock_index_pe_lg for {index_name}")
    
    date_col = pick_col(raw, ['日期', 'date', 'time'])
    if date_col is None:
        date_col = raw.columns[0]
    
    # Prioritize TTM PE (滚动市盈率), exclude static PE (静态市盈率)
    pe_col = pick_col(raw, ['滚动市盈率', '滚动'], exclude=['静态'])
    if pe_col is None:
        # Fallback: try TTM keywords with static exclusion
        pe_col = pick_col(raw, ['ttm', 'TTM', 'pe_ttm'], exclude=['静态', 'static'])
    if pe_col is None:
        # Last resort: any PE column excluding static
        pe_col = pick_col(raw, ['市盈率', 'pe', 'PE'], exclude=['静态', 'static', '静态市盈率'])
    if pe_col is None:
        # Manual search for TTM column
        for col in raw.columns:
            col_str = str(col)
            if '滚动' in col_str and '市盈率' in col_str:
                pe_col = col
                break
    if pe_col is None:
        logger.error(f"Available columns: {raw.columns.tolist()}")
        raise ValueError("Could not identify PE TTM column (滚动市盈率). Please check AkShare output.")
    
    logger.info(f"Using columns - Date: {date_col}, PE: {pe_col}")
    
    df = pd.DataFrame({
        'date': pd.to_datetime(raw[date_col], errors='coerce'),
        'pe_ttm': pd.to_numeric(raw[pe_col], errors='coerce')
    })
    
    df = df.dropna(subset=['date', 'pe_ttm'])
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    df = df.reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} days of PE data: {df['date'].min()} to {df['date'].max()}")
    return df[['date', 'pe_ttm']]


def fetch_index_prices(symbol: str = "000300") -> pd.DataFrame:
    """Fetch CSI300 index daily prices for return calculation."""
    import akshare as ak
    
    logger.info(f"Fetching index price data for {symbol}...")
    
    # Try different methods
    try:
        raw = retry_fetch(ak.stock_zh_index_daily, symbol=f"sh{symbol}")
    except:
        try:
            raw = retry_fetch(ak.index_zh_a_hist, symbol=symbol, period="daily", start_date="19900101")
        except:
            raw = retry_fetch(ak.stock_zh_index_daily_em, symbol=f"sh{symbol}")
    
    if raw is None or raw.empty:
        raise ValueError(f"Empty index price data for {symbol}")
    
    logger.debug(f"Index price columns: {raw.columns.tolist()}")
    
    date_col = pick_col(raw, ['日期', 'date', 'time'])
    if date_col is None:
        date_col = raw.columns[0]
    
    close_col = pick_col(raw, ['收盘', 'close', '收盘价'])
    if close_col is None:
        close_col = raw.columns[-1]  # Usually last column
    
    df = pd.DataFrame({
        'date': pd.to_datetime(raw[date_col], errors='coerce'),
        'close': pd.to_numeric(raw[close_col], errors='coerce')
    })
    
    df = df.dropna(subset=['date', 'close'])
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Loaded {len(df)} days of index prices: {df['date'].min()} to {df['date'].max()}")
    return df


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

@dataclass
class BacktestParams:
    """Parameter configuration for backtest."""
    dyn_window_months: int = 60
    dyn_quantile: float = 0.20
    pe_window_years: int = 10
    band: float = 0.05
    base_position: float = 0.50
    trend_sensitivity: float = 0.30
    cheap_threshold: float = 0.20
    expensive_threshold: float = 0.80
    max_weight: float = 1.2
    
    def to_dict(self) -> dict:
        return {
            'dyn_window_months': self.dyn_window_months,
            'dyn_quantile': self.dyn_quantile,
            'pe_window_years': self.pe_window_years,
            'band': self.band,
            'base_position': self.base_position,
            'trend_sensitivity': self.trend_sensitivity,
            'cheap_threshold': self.cheap_threshold,
            'expensive_threshold': self.expensive_threshold,
            'max_weight': self.max_weight
        }
    
    def to_label(self) -> str:
        return f"dw{self.dyn_window_months}_dq{self.dyn_quantile:.2f}_pw{self.pe_window_years}_b{self.band:.2f}_bp{self.base_position:.2f}"


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    params: BacktestParams
    period_years: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    
    # Performance metrics
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Benchmark comparison
    benchmark_return: float = 0.0
    benchmark_ann_return: float = 0.0
    benchmark_volatility: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_max_dd: float = 0.0
    excess_return: float = 0.0
    
    # Trade statistics
    num_trades: int = 0
    avg_position: float = 0.0
    
    # Monthly signals
    signals_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def to_dict(self) -> dict:
        return {
            **self.params.to_dict(),
            'period_years': self.period_years,
            'start_date': self.start_date.strftime('%Y-%m-%d'),
            'end_date': self.end_date.strftime('%Y-%m-%d'),
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'benchmark_return': self.benchmark_return,
            'benchmark_ann_return': self.benchmark_ann_return,
            'benchmark_volatility': self.benchmark_volatility,
            'benchmark_sharpe': self.benchmark_sharpe,
            'benchmark_max_dd': self.benchmark_max_dd,
            'excess_return': self.excess_return,
            'num_trades': self.num_trades,
            'avg_position': self.avg_position
        }


class BacktestEngine:
    """
    Backtesting engine for Strategy #1.
    
    Simulates the strategy month-by-month with no lookahead bias.
    """
    
    def __init__(
        self,
        macro_df: pd.DataFrame,
        pe_df: pd.DataFrame,
        price_df: pd.DataFrame,
        risk_free_rate: float = 0.02  # Annual risk-free rate
    ):
        self.macro_df = macro_df.copy()
        self.pe_df = pe_df.copy()
        self.price_df = price_df.copy()
        self.risk_free_rate = risk_free_rate
        
        # Precompute gap_ma3
        self.macro_df = compute_gap_ma3(self.macro_df)
        
        # Create monthly price series (month-end close)
        self.price_df['month_end'] = self.price_df['date'] + pd.offsets.MonthEnd(0)
        self.monthly_prices = self.price_df.groupby('month_end')['close'].last().reset_index()
        self.monthly_prices.columns = ['month_end', 'close']
        
        # Compute monthly returns
        self.monthly_prices['return'] = self.monthly_prices['close'].pct_change()
        
        logger.info(f"BacktestEngine initialized with {len(self.macro_df)} macro months, "
                   f"{len(self.pe_df)} PE days, {len(self.monthly_prices)} price months")
    
    def _get_available_months(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """Get list of months available for backtesting."""
        valid_macro = self.macro_df.dropna(subset=['gap_ma3'])
        
        # Need at least 2 months of gap_ma3 for trend comparison
        valid_months = valid_macro['month_end'].tolist()
        
        # Filter by date range
        months = [m for m in valid_months if start_date <= m <= end_date]
        
        # Exclude first month (need previous for trend)
        if len(months) > 1:
            months = months[1:]
        
        return sorted(months)
    
    def _simulate_month(
        self,
        month_end: pd.Timestamp,
        params: BacktestParams,
        last_target: Optional[float]
    ) -> Tuple[Optional[float], dict]:
        """
        Simulate strategy for a single month.
        
        IMPORTANT: Macro data has a 1-month publication lag.
        When deciding the position at the end of month T, we only use macro data
        available up to month T-1 (i.e., the macro observation for T-1 is the latest).
        
        Returns:
            Tuple of (target_position, signal_dict) or (None, {}) if cannot compute
        """
        # Apply 1-month publication lag for macro data
        # Position decided at end of month T uses macro data up to T-1
        macro_asof_month = month_end - pd.offsets.MonthEnd(1)
        
        # Get macro data up to T-1 (with publication lag)
        valid_macro = self.macro_df[
            (self.macro_df['month_end'] <= macro_asof_month) & 
            (self.macro_df['gap_ma3'].notna())
        ]
        
        if len(valid_macro) < 2:
            return None, {}
        
        # Current macro observation is T-1 (latest available with lag)
        current_macro_row = valid_macro.iloc[-1]
        current_gap_ma3 = current_macro_row['gap_ma3']
        current_macro_month = current_macro_row['month_end']
        
        # Previous macro observation is T-2
        prev_gap_ma3 = valid_macro.iloc[-2]['gap_ma3']
        
        # Dynamic threshold: computed using data up to T-2 (excluding T-1) to avoid lookahead
        # This ensures the threshold doesn't include the current observation being compared
        dyn_thr = compute_dynamic_threshold(
            self.macro_df, current_macro_month, 
            params.dyn_window_months, params.dyn_quantile,
            exclude_current=True  # Exclude current month from threshold calculation
        )
        
        # Trend signal based on lagged macro data
        trend_signal = compute_trend_signal(current_gap_ma3, prev_gap_ma3, dyn_thr)
        
        # PE percentile
        pe_asof = self.pe_df[self.pe_df['date'] <= month_end]['date'].max()
        if pd.isna(pe_asof):
            return None, {}
        
        pe_pct = pe_percentile_rolling(self.pe_df, pe_asof, params.pe_window_years)
        if pe_pct is None:
            return None, {}
        
        # Valuation weight
        val_weight = valuation_weight(
            pe_pct, 
            params.cheap_threshold, 
            params.expensive_threshold,
            params.max_weight
        )
        
        # Target position
        target = compute_target_position(
            trend_signal, val_weight,
            params.base_position, params.trend_sensitivity
        )
        
        # Trade band check
        traded = True
        if last_target is not None and params.band > 0:
            if abs(target - last_target) < params.band:
                target = last_target  # Keep old position
                traded = False
        
        signal = {
            'month_end': month_end,
            'macro_asof_month': current_macro_month,  # The macro month used (T-1 due to lag)
            'gap_ma3': current_gap_ma3,
            'prev_gap_ma3': prev_gap_ma3,
            'dyn_thr': dyn_thr,
            'trend_signal': trend_signal,
            'pe_pct': pe_pct,
            'val_weight': val_weight,
            'target_position': target,
            'traded': traded
        }
        
        return target, signal
    
    def run_backtest(
        self,
        params: BacktestParams,
        period_years: int,
        end_date: Optional[pd.Timestamp] = None
    ) -> BacktestResult:
        """
        Run backtest for given parameters and period.
        
        Args:
            params: Strategy parameters
            period_years: Lookback period in years
            end_date: End date for backtest (default: latest available)
            
        Returns:
            BacktestResult with performance metrics
        """
        if end_date is None:
            end_date = self.monthly_prices['month_end'].max()
        
        start_date = end_date - pd.DateOffset(years=period_years)
        
        # Get available months
        months = self._get_available_months(start_date, end_date)
        
        if len(months) < 12:
            logger.warning(f"Only {len(months)} months available for {period_years}Y backtest")
            return BacktestResult(params=params, period_years=period_years,
                                 start_date=start_date, end_date=end_date)
        
        # Simulate month by month
        signals = []
        positions = []
        last_target = None
        
        for month in months:
            target, signal = self._simulate_month(month, params, last_target)
            
            if target is not None:
                signals.append(signal)
                positions.append({'month_end': month, 'position': target})
                last_target = target
        
        if len(positions) < 12:
            logger.warning(f"Only {len(positions)} valid positions generated")
            return BacktestResult(params=params, period_years=period_years,
                                 start_date=start_date, end_date=end_date)
        
        # Convert to DataFrames
        signals_df = pd.DataFrame(signals)
        positions_df = pd.DataFrame(positions)
        
        # Merge with returns
        positions_df = positions_df.merge(
            self.monthly_prices[['month_end', 'return']], 
            on='month_end', how='left'
        )
        positions_df = positions_df.dropna(subset=['return'])
        
        # Calculate strategy returns (position * next month return)
        # Position at month-end M earns return during month M+1
        positions_df['strategy_return'] = positions_df['position'].shift(1) * positions_df['return']
        positions_df = positions_df.dropna(subset=['strategy_return'])
        
        if positions_df.empty:
            return BacktestResult(params=params, period_years=period_years,
                                 start_date=start_date, end_date=end_date)
        
        # Performance metrics
        result = self._compute_metrics(positions_df, signals_df, params, period_years, start_date, end_date)
        
        return result
    
    def _compute_metrics(
        self,
        positions_df: pd.DataFrame,
        signals_df: pd.DataFrame,
        params: BacktestParams,
        period_years: int,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp
    ) -> BacktestResult:
        """
        Compute performance metrics from positions and returns.
        
        Sharpe Ratio Calculation (standard monthly excess-return formulation):
        1. Compute monthly portfolio returns (position * index_return, no cash yield)
        2. Compute monthly risk-free rate: rf_monthly = (1 + annual_rf)^(1/12) - 1
        3. Compute monthly excess returns: strategy_return - rf_monthly
        4. Sharpe = mean(excess_returns) / std(excess_returns) * sqrt(12)
        """
        
        strategy_returns = positions_df['strategy_return']
        benchmark_returns = positions_df['return']
        
        n_months = len(strategy_returns)
        years = n_months / 12
        
        # Monthly risk-free rate
        rf_monthly = (1 + self.risk_free_rate) ** (1/12) - 1
        
        # Strategy metrics
        cum_return = (1 + strategy_returns).prod() - 1
        ann_return = (1 + cum_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = strategy_returns.std() * np.sqrt(12)
        
        # Standard Sharpe ratio: mean(excess_returns) / std(excess_returns) * sqrt(12)
        excess_returns = strategy_returns - rf_monthly
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(12)) if excess_returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_dd = drawdowns.min()
        
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0
        
        # Benchmark metrics (same Sharpe formulation)
        bench_cum = (1 + benchmark_returns).prod() - 1
        bench_ann = (1 + bench_cum) ** (1 / years) - 1 if years > 0 else 0
        bench_vol = benchmark_returns.std() * np.sqrt(12)
        
        bench_excess_returns = benchmark_returns - rf_monthly
        bench_sharpe = (bench_excess_returns.mean() / bench_excess_returns.std() * np.sqrt(12)) if bench_excess_returns.std() > 0 else 0
        
        bench_cumulative = (1 + benchmark_returns).cumprod()
        bench_rolling_max = bench_cumulative.expanding().max()
        bench_drawdowns = bench_cumulative / bench_rolling_max - 1
        bench_max_dd = bench_drawdowns.min()
        
        # Trade statistics
        num_trades = signals_df['traded'].sum() if 'traded' in signals_df.columns else 0
        avg_position = positions_df['position'].mean()
        
        return BacktestResult(
            params=params,
            period_years=period_years,
            start_date=start_date,
            end_date=end_date,
            total_return=cum_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            benchmark_return=bench_cum,
            benchmark_ann_return=bench_ann,
            benchmark_volatility=bench_vol,
            benchmark_sharpe=bench_sharpe,
            benchmark_max_dd=bench_max_dd,
            excess_return=ann_return - bench_ann,
            num_trades=int(num_trades),
            avg_position=avg_position,
            signals_df=signals_df
        )


# =============================================================================
# PARAMETER GRID SEARCH
# =============================================================================

def generate_param_grid(
    dyn_window_months: List[int] = [36, 48, 60, 72],
    dyn_quantile: List[float] = [0.15, 0.20, 0.25],
    pe_window_years: List[int] = [5, 7, 10],
    band: List[float] = [0.0, 0.05, 0.10],
    base_position: List[float] = [0.30, 0.40, 0.50, 0.60, 0.70],
    cheap_threshold: List[float] = [0.20],
    expensive_threshold: List[float] = [0.80],
) -> List[BacktestParams]:
    """Generate parameter combinations for grid search."""
    
    combinations = itertools.product(
        dyn_window_months,
        dyn_quantile,
        pe_window_years,
        band,
        base_position,
        cheap_threshold,
        expensive_threshold
    )
    
    params_list = []
    for dw, dq, pw, b, bp, ct, et in combinations:
        params_list.append(BacktestParams(
            dyn_window_months=dw,
            dyn_quantile=dq,
            pe_window_years=pw,
            band=b,
            base_position=bp,
            cheap_threshold=ct,
            expensive_threshold=et
        ))
    
    return params_list


def run_grid_search(
    engine: BacktestEngine,
    params_grid: List[BacktestParams],
    periods: List[int] = [10, 20]
) -> pd.DataFrame:
    """
    Run backtests for all parameter combinations and periods.
    
    Returns:
        DataFrame with all results
    """
    results = []
    total = len(params_grid) * len(periods)
    
    logger.info(f"Starting grid search: {len(params_grid)} param sets × {len(periods)} periods = {total} runs")
    
    for i, params in enumerate(params_grid):
        for period in periods:
            try:
                result = engine.run_backtest(params, period)
                results.append(result.to_dict())
                
                if (i * len(periods) + periods.index(period) + 1) % 10 == 0:
                    logger.info(f"Progress: {i * len(periods) + periods.index(period) + 1}/{total}")
                    
            except Exception as e:
                logger.warning(f"Failed for {params.to_label()}, {period}Y: {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# REPORTING
# =============================================================================

def generate_summary_report(results_df: pd.DataFrame, outdir: Path) -> None:
    """Generate summary statistics and charts."""
    
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    results_df.to_csv(outdir / 'backtest_results_full.csv', index=False)
    logger.info(f"Saved full results to {outdir / 'backtest_results_full.csv'}")
    
    # Summary by period
    for period in results_df['period_years'].unique():
        period_df = results_df[results_df['period_years'] == period].copy()
        
        # Sort by Sharpe ratio
        period_df = period_df.sort_values('sharpe_ratio', ascending=False)
        
        # Save period results
        period_df.to_csv(outdir / f'backtest_results_{period}Y.csv', index=False)
        
        # Print top configurations
        print(f"\n{'='*80}")
        print(f"TOP 10 CONFIGURATIONS - {period} YEAR BACKTEST")
        print(f"{'='*80}")
        
        top10 = period_df.head(10)
        for idx, row in top10.iterrows():
            print(f"\nRank {top10.index.get_loc(idx) + 1}:")
            print(f"  Params: dyn_window={row['dyn_window_months']}m, "
                  f"dyn_quantile={row['dyn_quantile']:.2f}, "
                  f"pe_window={row['pe_window_years']}y, "
                  f"band={row['band']:.2f}, "
                  f"base_pos={row['base_position']:.2f}")
            print(f"  Sharpe Ratio: {row['sharpe_ratio']:.3f}")
            print(f"  Ann. Return:  {row['annualized_return']:.2%}")
            print(f"  Volatility:   {row['volatility']:.2%}")
            print(f"  Max Drawdown: {row['max_drawdown']:.2%}")
            print(f"  Excess Return:{row['excess_return']:.2%} vs benchmark")
        
        # Benchmark comparison
        bench_row = period_df.iloc[0]
        print(f"\nBENCHMARK ({period}Y Buy-and-Hold):")
        print(f"  Ann. Return:  {bench_row['benchmark_ann_return']:.2%}")
        print(f"  Volatility:   {bench_row['benchmark_volatility']:.2%}")
        print(f"  Sharpe Ratio: {bench_row['benchmark_sharpe']:.3f}")
        print(f"  Max Drawdown: {bench_row['benchmark_max_dd']:.2%}")
    
    # Parameter sensitivity analysis
    print(f"\n{'='*80}")
    print("PARAMETER SENSITIVITY ANALYSIS")
    print(f"{'='*80}")
    
    for param in ['dyn_window_months', 'dyn_quantile', 'pe_window_years', 'band', 'base_position']:
        print(f"\n{param}:")
        sensitivity = results_df.groupby(param).agg({
            'sharpe_ratio': ['mean', 'std'],
            'annualized_return': 'mean',
            'max_drawdown': 'mean'
        }).round(4)
        print(sensitivity.to_string())
    
    # Generate summary statistics CSV
    summary_stats = results_df.groupby('period_years').agg({
        'sharpe_ratio': ['mean', 'std', 'min', 'max'],
        'annualized_return': ['mean', 'std', 'min', 'max'],
        'max_drawdown': ['mean', 'min'],
        'excess_return': ['mean', 'std']
    }).round(4)
    
    summary_stats.to_csv(outdir / 'backtest_summary_stats.csv')
    logger.info(f"Saved summary stats to {outdir / 'backtest_summary_stats.csv'}")


def generate_equity_curve(
    engine: BacktestEngine,
    params: BacktestParams,
    period_years: int,
    outdir: Path
) -> None:
    """Generate equity curve data for the best configuration."""
    
    result = engine.run_backtest(params, period_years)
    
    if result.signals_df.empty:
        logger.warning("No signals to plot")
        return
    
    # Save signals
    result.signals_df.to_csv(outdir / f'signals_{params.to_label()}_{period_years}Y.csv', index=False)
    logger.info(f"Saved signals to {outdir / f'signals_{params.to_label()}_{period_years}Y.csv'}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest Strategy #1: Macro Liquidity + CSI300 Valuation"
    )
    parser.add_argument(
        '--index', type=str, default='沪深300',
        help='Index name for PE data (default: 沪深300)'
    )
    parser.add_argument(
        '--index_code', type=str, default='000300',
        help='Index code for price data (default: 000300)'
    )
    parser.add_argument(
        '--periods', type=int, nargs='+', default=[10, 20],
        help='Backtest periods in years (default: 10 20)'
    )
    parser.add_argument(
        '--outdir', type=str, default='./backtest_results',
        help='Output directory (default: ./backtest_results)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick test with reduced parameter grid'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    logger.info("Fetching historical data...")
    try:
        macro_df = fetch_macro_money_supply()
        pe_df = fetch_csi300_pe(args.index)
        price_df = fetch_index_prices(args.index_code)
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        sys.exit(1)
    
    # Save raw data
    macro_df.to_csv(outdir / 'data_macro.csv', index=False)
    pe_df.to_csv(outdir / 'data_pe.csv', index=False)
    price_df.to_csv(outdir / 'data_prices.csv', index=False)
    logger.info("Saved raw data to output directory")
    
    # Initialize backtest engine
    engine = BacktestEngine(macro_df, pe_df, price_df)
    
    # Generate parameter grid
    if args.quick:
        params_grid = generate_param_grid(
            dyn_window_months=[48, 60],
            dyn_quantile=[0.20],
            pe_window_years=[7, 10],
            band=[0.0, 0.05],
            base_position=[0.40, 0.50, 0.60]
        )
    else:
        params_grid = generate_param_grid(
            dyn_window_months=[36, 48, 60, 72, 84],
            dyn_quantile=[0.10, 0.15, 0.20, 0.25, 0.30],
            pe_window_years=[5, 7, 10, 12],
            band=[0.0, 0.03, 0.05, 0.08, 0.10],
            base_position=[0.30, 0.40, 0.50, 0.60, 0.70],
            cheap_threshold=[0.15, 0.20, 0.25],
            expensive_threshold=[0.75, 0.80, 0.85]
        )
    
    logger.info(f"Generated {len(params_grid)} parameter configurations")
    
    # Run grid search
    results_df = run_grid_search(engine, params_grid, args.periods)
    
    # Generate reports
    generate_summary_report(results_df, outdir)
    
    # Generate equity curves for best configs
    for period in args.periods:
        period_results = results_df[results_df['period_years'] == period]
        if not period_results.empty:
            best_row = period_results.loc[period_results['sharpe_ratio'].idxmax()]
            best_params = BacktestParams(
                dyn_window_months=int(best_row['dyn_window_months']),
                dyn_quantile=float(best_row['dyn_quantile']),
                pe_window_years=int(best_row['pe_window_years']),
                band=float(best_row['band']),
                base_position=float(best_row['base_position']),
                cheap_threshold=float(best_row['cheap_threshold']),
                expensive_threshold=float(best_row['expensive_threshold'])
            )
            generate_equity_curve(engine, best_params, period, outdir)
    
    logger.info(f"Backtest complete. Results saved to {outdir}")


if __name__ == "__main__":
    main()