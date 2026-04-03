#!/usr/bin/env python3
"""
Backtest Framework v2: Macro Liquidity + Valuation Strategy

METHODOLOGY FIXES:
1. Cash yield: r_p = pos * r_eq + (1-pos) * r_cash
2. Transaction costs: cost_rate * |pos_t - pos_{t-1}|
3. No hard-coded fallbacks; expanding window warm-up
4. Configurable valuation floor (min_val_weight)
5. Proper data alignment (month-end signals → next month returns)

EXTENSIONS:
- Multi-index: CSI300, CSI500, CSI1000
- Benchmarks: Buy&Hold, Fixed-mix, Vol-matched
- Excel output with per-index and overall sheets

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
from typing import Dict, List, Optional, Tuple, Any, Union

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
# CONFIGURATION
# =============================================================================

# Index configurations
INDEX_CONFIG = {
    'CSI300': {
        'name': '沪深300',
        'code': '000300',
        'etf': '510300'
    },
    'CSI500': {
        'name': '中证500',
        'code': '000905',
        'etf': '510500'
    },
    'CSI1000': {
        'name': '中证1000',
        'code': '000852',
        'etf': '512100'
    }
}

# Default annual risk-free rate (used for cash yield and Sharpe)
DEFAULT_RF_ANNUAL = 0.02

# Default transaction cost (one-way, as fraction)
DEFAULT_COST_RATE = 0.001  # 10 bps


# =============================================================================
# UTILITY FUNCTIONS
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


def compute_monthly_rf(annual_rf: float) -> float:
    """Convert annual risk-free rate to monthly."""
    return (1 + annual_rf) ** (1/12) - 1


# =============================================================================
# SIGNAL COMPUTATION (with warm-up handling)
# =============================================================================

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
    min_history: int = 24
) -> Optional[float]:
    """
    Compute dynamic threshold as percentile of recent gap_ma3 values.
    
    NO HARD-CODED FALLBACK: Returns None if insufficient history.
    
    Args:
        macro_df: DataFrame with 'month_end' and 'gap_ma3' columns
        asof_month: Reference month for lookback (exclusive - data up to asof_month-1)
        window_months: Number of months to look back
        quantile: Percentile threshold
        min_history: Minimum months required (no fallback if not met)
    
    Returns:
        Dynamic threshold value or None if insufficient data
    """
    # Use data strictly before asof_month (no lookahead)
    mask = macro_df['month_end'] < asof_month
    recent = macro_df.loc[mask, 'gap_ma3'].dropna().tail(window_months)
    
    if len(recent) < min_history:
        return None  # No hard-coded fallback
    
    return float(np.percentile(recent, quantile * 100))


def compute_trend_signal(
    current_gap_ma3: float,
    prev_gap_ma3: float,
    dyn_threshold: Optional[float]
) -> int:
    """
    Compute trend signal based on liquidity conditions.
    
    If dyn_threshold is None (warm-up period), returns 0 (neutral).
    """
    if dyn_threshold is None:
        return 0  # Neutral during warm-up
    
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
    window_years: int = 10,
    min_obs: int = 252  # ~1 year minimum
) -> Optional[float]:
    """
    Compute rolling PE percentile with no lookahead bias.
    
    Returns None if insufficient history (no fallback).
    """
    mask = pe_df['date'] <= asof_date
    available = pe_df.loc[mask].copy()
    
    if available.empty:
        return None
    
    current_pe = available['pe_ttm'].iloc[-1]
    current_date = available['date'].iloc[-1]
    
    window_start = current_date - pd.DateOffset(years=window_years)
    hist_mask = (available['date'] >= window_start) & (available['date'] <= current_date)
    hist_pe = available.loc[hist_mask, 'pe_ttm']
    
    if len(hist_pe) < min_obs:
        return None  # No fallback
    
    return float((hist_pe <= current_pe).mean())


def valuation_weight(
    pe_pct: Optional[float],
    cheap_threshold: float = 0.20,
    expensive_threshold: float = 0.80,
    max_weight: float = 1.2,
    min_weight: float = 0.2  # NEW: Configurable floor (not zero)
) -> float:
    """
    Compute valuation weight from PE percentile with configurable floor.
    
    Piecewise linear mapping:
    - pe_pct <= cheap_threshold → max_weight
    - pe_pct >= expensive_threshold → min_weight (not zero!)
    - in between → linear interpolation
    
    If pe_pct is None (warm-up), returns (max_weight + min_weight) / 2.
    """
    if pe_pct is None:
        return (max_weight + min_weight) / 2  # Neutral during warm-up
    
    if pe_pct <= cheap_threshold:
        return max_weight
    elif pe_pct >= expensive_threshold:
        return min_weight
    else:
        # Linear interpolation from max_weight to min_weight
        slope = (min_weight - max_weight) / (expensive_threshold - cheap_threshold)
        return max_weight + slope * (pe_pct - cheap_threshold)


def compute_target_position(
    trend_signal: int,
    val_weight: float,
    base_position: float = 0.50,
    trend_sensitivity: float = 0.30
) -> float:
    """Compute target position from signals."""
    raw_target = (base_position + trend_sensitivity * trend_signal) * val_weight
    return float(np.clip(raw_target, 0.0, 1.0))


# =============================================================================
# DATA FETCHING (with caching)
# =============================================================================

_DATA_CACHE = {}

def fetch_macro_money_supply(use_cache: bool = True) -> pd.DataFrame:
    """Fetch M1/M2 YoY data from AkShare with caching."""
    cache_key = 'macro_money_supply'
    if use_cache and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()
    
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
    
    result = df[['month_end', 'm1_yoy', 'm2_yoy', 'gap']]
    
    if use_cache:
        _DATA_CACHE[cache_key] = result.copy()
    
    logger.info(f"Loaded {len(result)} months of macro data: {result['month_end'].min()} to {result['month_end'].max()}")
    return result


def fetch_index_pe(index_name: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch index PE(TTM) data from AkShare with caching."""
    cache_key = f'pe_{index_name}'
    if use_cache and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()
    
    import akshare as ak
    
    logger.info(f"Fetching PE data for {index_name}...")
    raw = retry_fetch(ak.stock_index_pe_lg, symbol=index_name)
    
    if raw is None or raw.empty:
        raise ValueError(f"Empty data from stock_index_pe_lg for {index_name}")
    
    date_col = pick_col(raw, ['日期', 'date', 'time'])
    if date_col is None:
        date_col = raw.columns[0]
    
    # Prioritize TTM PE, exclude static PE
    pe_col = pick_col(raw, ['滚动市盈率', '滚动'], exclude=['静态'])
    if pe_col is None:
        pe_col = pick_col(raw, ['ttm', 'TTM', 'pe_ttm'], exclude=['静态', 'static'])
    if pe_col is None:
        pe_col = pick_col(raw, ['市盈率', 'pe', 'PE'], exclude=['静态', 'static', '静态市盈率'])
    if pe_col is None:
        for col in raw.columns:
            col_str = str(col)
            if '滚动' in col_str and '市盈率' in col_str:
                pe_col = col
                break
    if pe_col is None:
        logger.error(f"Available columns: {raw.columns.tolist()}")
        raise ValueError(f"Could not identify PE TTM column for {index_name}")
    
    logger.info(f"Using columns - Date: {date_col}, PE: {pe_col}")
    
    df = pd.DataFrame({
        'date': pd.to_datetime(raw[date_col], errors='coerce'),
        'pe_ttm': pd.to_numeric(raw[pe_col], errors='coerce')
    })
    
    df = df.dropna(subset=['date', 'pe_ttm'])
    df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')
    df = df.reset_index(drop=True)
    
    result = df[['date', 'pe_ttm']]
    
    if use_cache:
        _DATA_CACHE[cache_key] = result.copy()
    
    logger.info(f"Loaded {len(result)} days of PE data: {result['date'].min()} to {result['date'].max()}")
    return result


def fetch_index_prices(index_code: str, use_cache: bool = True) -> pd.DataFrame:
    """Fetch index daily prices for return calculation with caching."""
    cache_key = f'prices_{index_code}'
    if use_cache and cache_key in _DATA_CACHE:
        return _DATA_CACHE[cache_key].copy()
    
    import akshare as ak
    
    logger.info(f"Fetching index price data for {index_code}...")
    
    raw = None
    # Try different methods
    try:
        raw = retry_fetch(ak.stock_zh_index_daily, symbol=f"sh{index_code}")
    except:
        pass
    
    if raw is None or raw.empty:
        try:
            raw = retry_fetch(ak.index_zh_a_hist, symbol=index_code, period="daily", start_date="19900101")
        except:
            pass
    
    if raw is None or raw.empty:
        try:
            raw = retry_fetch(ak.stock_zh_index_daily_em, symbol=f"sh{index_code}")
        except:
            pass
    
    if raw is None or raw.empty:
        raise ValueError(f"Could not fetch index price data for {index_code}")
    
    date_col = pick_col(raw, ['日期', 'date', 'time'])
    if date_col is None:
        date_col = raw.columns[0]
    
    close_col = pick_col(raw, ['收盘', 'close', '收盘价'])
    if close_col is None:
        close_col = raw.columns[-1]
    
    df = pd.DataFrame({
        'date': pd.to_datetime(raw[date_col], errors='coerce'),
        'close': pd.to_numeric(raw[close_col], errors='coerce')
    })
    
    df = df.dropna(subset=['date', 'close'])
    df = df.sort_values('date').reset_index(drop=True)
    
    result = df[['date', 'close']]
    
    if use_cache:
        _DATA_CACHE[cache_key] = result.copy()
    
    logger.info(f"Loaded {len(result)} days of index prices: {result['date'].min()} to {result['date'].max()}")
    return result


# =============================================================================
# BACKTEST PARAMETERS AND RESULTS
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
    min_weight: float = 0.2  # NEW: Valuation floor
    cost_rate: float = 0.001  # NEW: Transaction cost rate
    
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
            'max_weight': self.max_weight,
            'min_weight': self.min_weight,
            'cost_rate': self.cost_rate
        }
    
    def to_label(self) -> str:
        return (f"dw{self.dyn_window_months}_dq{self.dyn_quantile:.2f}_"
                f"pw{self.pe_window_years}_bp{self.base_position:.2f}_"
                f"minw{self.min_weight:.2f}_cost{self.cost_rate:.4f}")


@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    params: BacktestParams
    index_name: str
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
    
    # NEW: Additional metrics
    annualized_excess: float = 0.0
    avg_position: float = 0.0
    turnover: float = 0.0  # Total turnover over period
    cost_drag: float = 0.0  # Annualized cost drag
    rebalance_count: int = 0
    
    # Benchmark comparison
    benchmark_return: float = 0.0
    benchmark_ann_return: float = 0.0
    benchmark_volatility: float = 0.0
    benchmark_sharpe: float = 0.0
    benchmark_max_dd: float = 0.0
    
    # Fixed-mix benchmark
    fixedmix_ann_return: float = 0.0
    fixedmix_sharpe: float = 0.0
    fixedmix_max_dd: float = 0.0
    
    # Vol-matched benchmark
    volmatch_ann_return: float = 0.0
    volmatch_sharpe: float = 0.0
    
    # Monthly data (for analysis)
    monthly_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def to_dict(self) -> dict:
        return {
            **self.params.to_dict(),
            'index_name': self.index_name,
            'period_years': self.period_years,
            'start_date': self.start_date.strftime('%Y-%m-%d') if pd.notna(self.start_date) else '',
            'end_date': self.end_date.strftime('%Y-%m-%d') if pd.notna(self.end_date) else '',
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'calmar_ratio': self.calmar_ratio,
            'annualized_excess': self.annualized_excess,
            'avg_position': self.avg_position,
            'turnover': self.turnover,
            'cost_drag': self.cost_drag,
            'rebalance_count': self.rebalance_count,
            'benchmark_ann_return': self.benchmark_ann_return,
            'benchmark_volatility': self.benchmark_volatility,
            'benchmark_sharpe': self.benchmark_sharpe,
            'benchmark_max_dd': self.benchmark_max_dd,
            'fixedmix_ann_return': self.fixedmix_ann_return,
            'fixedmix_sharpe': self.fixedmix_sharpe,
            'fixedmix_max_dd': self.fixedmix_max_dd,
            'volmatch_ann_return': self.volmatch_ann_return,
            'volmatch_sharpe': self.volmatch_sharpe
        }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Backtesting engine with proper methodology.
    
    Features:
    - Cash yield on uninvested portion
    - Transaction costs
    - No hard-coded fallbacks
    - Multiple benchmarks
    """
    
    def __init__(
        self,
        macro_df: pd.DataFrame,
        pe_df: pd.DataFrame,
        price_df: pd.DataFrame,
        index_name: str = "CSI300",
        risk_free_rate: float = DEFAULT_RF_ANNUAL,
        fixed_mix_equity: float = 0.6  # For 60/40 benchmark
    ):
        self.macro_df = macro_df.copy()
        self.pe_df = pe_df.copy()
        self.price_df = price_df.copy()
        self.index_name = index_name
        self.risk_free_rate = risk_free_rate
        self.rf_monthly = compute_monthly_rf(risk_free_rate)
        self.fixed_mix_equity = fixed_mix_equity
        
        # Precompute gap_ma3
        self.macro_df = compute_gap_ma3(self.macro_df)
        
        # Create monthly price series (month-end close)
        self.price_df['month_end'] = self.price_df['date'] + pd.offsets.MonthEnd(0)
        self.monthly_prices = self.price_df.groupby('month_end')['close'].last().reset_index()
        self.monthly_prices.columns = ['month_end', 'close']
        
        # Compute monthly equity returns
        self.monthly_prices['eq_return'] = self.monthly_prices['close'].pct_change()
        
        logger.info(f"BacktestEngine initialized for {index_name}: "
                   f"{len(self.macro_df)} macro months, "
                   f"{len(self.pe_df)} PE days, "
                   f"{len(self.monthly_prices)} price months")
    
    def _get_available_months(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> List[pd.Timestamp]:
        """Get list of months available for backtesting."""
        # Need valid macro data AND price data
        valid_macro = self.macro_df.dropna(subset=['gap_ma3'])
        macro_months = set(valid_macro['month_end'].tolist())
        price_months = set(self.monthly_prices['month_end'].tolist())
        
        valid_months = sorted(macro_months & price_months)
        
        # Filter by date range
        months = [m for m in valid_months if start_date <= m <= end_date]
        
        # Need at least 2 months (for trend comparison)
        if len(months) > 1:
            months = months[1:]
        
        return months
    
    def _simulate_month(
        self,
        month_end: pd.Timestamp,
        params: BacktestParams,
        last_target: Optional[float]
    ) -> Tuple[Optional[float], dict]:
        """
        Simulate strategy for a single month.
        
        1-month publication lag: position at end of T uses macro data through T-1.
        """
        # Apply 1-month publication lag
        macro_asof_month = month_end - pd.offsets.MonthEnd(1)
        
        # Get macro data up to T-1
        valid_macro = self.macro_df[
            (self.macro_df['month_end'] <= macro_asof_month) & 
            (self.macro_df['gap_ma3'].notna())
        ]
        
        if len(valid_macro) < 2:
            return None, {}
        
        # Current macro observation is T-1 (latest with lag)
        current_macro_row = valid_macro.iloc[-1]
        current_gap_ma3 = current_macro_row['gap_ma3']
        current_macro_month = current_macro_row['month_end']
        
        # Previous macro observation is T-2
        prev_gap_ma3 = valid_macro.iloc[-2]['gap_ma3']
        
        # Dynamic threshold (no fallback - returns None if insufficient history)
        dyn_thr = compute_dynamic_threshold(
            self.macro_df, current_macro_month,
            params.dyn_window_months, params.dyn_quantile,
            min_history=24
        )
        
        # Trend signal (neutral if dyn_thr is None)
        trend_signal = compute_trend_signal(current_gap_ma3, prev_gap_ma3, dyn_thr)
        
        # PE percentile
        pe_asof = self.pe_df[self.pe_df['date'] <= month_end]['date'].max()
        if pd.isna(pe_asof):
            pe_pct = None
        else:
            pe_pct = pe_percentile_rolling(self.pe_df, pe_asof, params.pe_window_years)
        
        # Valuation weight (with floor, neutral if pe_pct is None)
        val_weight = valuation_weight(
            pe_pct,
            params.cheap_threshold,
            params.expensive_threshold,
            params.max_weight,
            params.min_weight
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
                target = last_target
                traded = False
        
        signal = {
            'month_end': month_end,
            'macro_asof_month': current_macro_month,
            'gap_ma3': current_gap_ma3,
            'prev_gap_ma3': prev_gap_ma3,
            'dyn_thr': dyn_thr,
            'trend_signal': trend_signal,
            'pe_pct': pe_pct,
            'val_weight': val_weight,
            'target_position': target,
            'traded': traded,
            'in_warmup': (dyn_thr is None or pe_pct is None)
        }
        
        return target, signal
    
    def run_backtest(
        self,
        params: BacktestParams,
        period_years: int,
        end_date: Optional[pd.Timestamp] = None
    ) -> BacktestResult:
        """Run backtest with proper methodology."""
        
        if end_date is None:
            end_date = self.monthly_prices['month_end'].max()
        
        start_date = end_date - pd.DateOffset(years=period_years)
        
        # Get available months
        months = self._get_available_months(start_date, end_date)
        
        if len(months) < 12:
            logger.warning(f"Only {len(months)} months available for {period_years}Y backtest")
            return BacktestResult(
                params=params, index_name=self.index_name,
                period_years=period_years, start_date=start_date, end_date=end_date
            )
        
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
            return BacktestResult(
                params=params, index_name=self.index_name,
                period_years=period_years, start_date=start_date, end_date=end_date
            )
        
        # Build monthly DataFrame
        positions_df = pd.DataFrame(positions)
        signals_df = pd.DataFrame(signals)
        
        # Merge with returns
        positions_df = positions_df.merge(
            self.monthly_prices[['month_end', 'eq_return']],
            on='month_end', how='left'
        )
        positions_df = positions_df.dropna(subset=['eq_return'])
        
        if positions_df.empty:
            return BacktestResult(
                params=params, index_name=self.index_name,
                period_years=period_years, start_date=start_date, end_date=end_date
            )
        
        # Calculate position changes for turnover and costs
        positions_df['prev_position'] = positions_df['position'].shift(1).fillna(0)
        positions_df['position_change'] = (positions_df['position'] - positions_df['prev_position']).abs()
        
        # Portfolio return: pos * r_eq + (1-pos) * r_cash
        # Position at month-end M earns return during month M+1
        positions_df['pos_lagged'] = positions_df['position'].shift(1)
        positions_df = positions_df.dropna(subset=['pos_lagged'])
        
        # Cash return (monthly risk-free)
        positions_df['cash_return'] = self.rf_monthly
        
        # Portfolio return BEFORE costs
        positions_df['port_return_gross'] = (
            positions_df['pos_lagged'] * positions_df['eq_return'] +
            (1 - positions_df['pos_lagged']) * positions_df['cash_return']
        )
        
        # Transaction costs (deducted from monthly return)
        positions_df['tx_cost'] = params.cost_rate * positions_df['position_change']
        positions_df['port_return'] = positions_df['port_return_gross'] - positions_df['tx_cost']
        
        # Compute metrics
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
        """Compute all performance metrics."""
        
        strategy_returns = positions_df['port_return']
        equity_returns = positions_df['eq_return']
        
        n_months = len(strategy_returns)
        years = n_months / 12
        
        # ===== Strategy Metrics =====
        cum_return = (1 + strategy_returns).prod() - 1
        ann_return = (1 + cum_return) ** (1 / years) - 1 if years > 0 else 0
        volatility = strategy_returns.std() * np.sqrt(12)
        
        # Sharpe: excess over cash, consistent with portfolio return formula
        excess_returns = strategy_returns - self.rf_monthly
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(12)) if excess_returns.std() > 0 else 0
        
        # Max drawdown
        cumulative = (1 + strategy_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = cumulative / rolling_max - 1
        max_dd = drawdowns.min()
        
        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0
        
        # Turnover and cost metrics
        total_turnover = positions_df['position_change'].sum()
        total_cost = positions_df['tx_cost'].sum()
        cost_drag = total_cost / years if years > 0 else 0
        rebalance_count = (positions_df['position_change'] > 1e-6).sum()
        avg_position = positions_df['pos_lagged'].mean()
        
        # ===== 100% Equity Benchmark =====
        bench_cum = (1 + equity_returns).prod() - 1
        bench_ann = (1 + bench_cum) ** (1 / years) - 1 if years > 0 else 0
        bench_vol = equity_returns.std() * np.sqrt(12)
        bench_excess = equity_returns - self.rf_monthly
        bench_sharpe = (bench_excess.mean() / bench_excess.std() * np.sqrt(12)) if bench_excess.std() > 0 else 0
        
        bench_cumulative = (1 + equity_returns).cumprod()
        bench_rolling_max = bench_cumulative.expanding().max()
        bench_drawdowns = bench_cumulative / bench_rolling_max - 1
        bench_max_dd = bench_drawdowns.min()
        
        # ===== Fixed-Mix Benchmark (e.g., 60/40) =====
        fixedmix_returns = (
            self.fixed_mix_equity * equity_returns +
            (1 - self.fixed_mix_equity) * self.rf_monthly
        )
        fixedmix_cum = (1 + fixedmix_returns).prod() - 1
        fixedmix_ann = (1 + fixedmix_cum) ** (1 / years) - 1 if years > 0 else 0
        fixedmix_excess = fixedmix_returns - self.rf_monthly
        fixedmix_sharpe = (fixedmix_excess.mean() / fixedmix_excess.std() * np.sqrt(12)) if fixedmix_excess.std() > 0 else 0
        
        fixedmix_cumulative = (1 + fixedmix_returns).cumprod()
        fixedmix_rolling_max = fixedmix_cumulative.expanding().max()
        fixedmix_drawdowns = fixedmix_cumulative / fixedmix_rolling_max - 1
        fixedmix_max_dd = fixedmix_drawdowns.min()
        
        # ===== Vol-Matched Benchmark =====
        # Scale benchmark exposure to match strategy realized volatility
        strategy_vol = strategy_returns.std()
        bench_vol_monthly = equity_returns.std()
        
        if bench_vol_monthly > 0:
            vol_scale = strategy_vol / bench_vol_monthly
            vol_scale = min(vol_scale, 1.5)  # Cap at 150% exposure
            volmatch_returns = vol_scale * equity_returns + (1 - vol_scale) * self.rf_monthly
        else:
            volmatch_returns = equity_returns
        
        volmatch_cum = (1 + volmatch_returns).prod() - 1
        volmatch_ann = (1 + volmatch_cum) ** (1 / years) - 1 if years > 0 else 0
        volmatch_excess = volmatch_returns - self.rf_monthly
        volmatch_sharpe = (volmatch_excess.mean() / volmatch_excess.std() * np.sqrt(12)) if volmatch_excess.std() > 0 else 0
        
        # Annualized excess return vs cash
        ann_excess = ann_return - self.risk_free_rate
        
        return BacktestResult(
            params=params,
            index_name=self.index_name,
            period_years=period_years,
            start_date=positions_df['month_end'].min(),
            end_date=positions_df['month_end'].max(),
            total_return=cum_return,
            annualized_return=ann_return,
            volatility=volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            annualized_excess=ann_excess,
            avg_position=avg_position,
            turnover=total_turnover,
            cost_drag=cost_drag,
            rebalance_count=int(rebalance_count),
            benchmark_return=bench_cum,
            benchmark_ann_return=bench_ann,
            benchmark_volatility=bench_vol,
            benchmark_sharpe=bench_sharpe,
            benchmark_max_dd=bench_max_dd,
            fixedmix_ann_return=fixedmix_ann,
            fixedmix_sharpe=fixedmix_sharpe,
            fixedmix_max_dd=fixedmix_max_dd,
            volmatch_ann_return=volmatch_ann,
            volmatch_sharpe=volmatch_sharpe,
            monthly_df=positions_df
        )


# =============================================================================
# PARAMETER GRID
# =============================================================================

def generate_param_grid(
    dyn_window_months: List[int] = [24, 36, 48, 60, 72, 84],
    dyn_quantile: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30],
    pe_window_years: List[int] = [3, 5, 7, 10, 12],
    band: List[float] = [0.0, 0.02, 0.03, 0.05, 0.08, 0.10],
    base_position: List[float] = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
    trend_sensitivity: List[float] = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    cheap_threshold: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30],
    expensive_threshold: List[float] = [0.75, 0.80, 0.85, 0.90, 0.95],
    max_weight: List[float] = [1.0, 1.2, 1.4],
    min_weight: List[float] = [0.0, 0.1, 0.2, 0.3],
    cost_rate: List[float] = [0.001]
) -> List[BacktestParams]:
    """Generate parameter combinations for grid search."""
    
    combinations = itertools.product(
        dyn_window_months,
        dyn_quantile,
        pe_window_years,
        band,
        base_position,
        trend_sensitivity,
        cheap_threshold,
        expensive_threshold,
        max_weight,
        min_weight,
        cost_rate
    )
    
    params_list = []
    for dw, dq, pw, b, bp, ts, ct, et, mxw, mnw, cr in combinations:
        params_list.append(BacktestParams(
            dyn_window_months=dw,
            dyn_quantile=dq,
            pe_window_years=pw,
            band=b,
            base_position=bp,
            trend_sensitivity=ts,
            cheap_threshold=ct,
            expensive_threshold=et,
            max_weight=mxw,
            min_weight=mnw,
            cost_rate=cr
        ))
    
    return params_list


def run_grid_search(
    engine: BacktestEngine,
    params_grid: List[BacktestParams],
    periods: List[int] = [10, 20]
) -> pd.DataFrame:
    """Run backtests for all parameter combinations and periods."""
    
    results = []
    total = len(params_grid) * len(periods)
    
    logger.info(f"Starting grid search for {engine.index_name}: "
               f"{len(params_grid)} params × {len(periods)} periods = {total} runs")
    
    for i, params in enumerate(params_grid):
        for period in periods:
            try:
                result = engine.run_backtest(params, period)
                results.append(result.to_dict())
                
                if (i * len(periods) + periods.index(period) + 1) % 50 == 0:
                    logger.info(f"Progress: {i * len(periods) + periods.index(period) + 1}/{total}")
                    
            except Exception as e:
                logger.warning(f"Failed for {params.to_label()}, {period}Y: {e}")
    
    return pd.DataFrame(results)


# =============================================================================
# MULTI-INDEX BACKTEST
# =============================================================================

def run_multi_index_backtest(
    indices: List[str],
    params_grid: List[BacktestParams],
    periods: List[int],
    risk_free_rate: float = DEFAULT_RF_ANNUAL,
    fixed_mix_equity: float = 0.6
) -> Dict[str, pd.DataFrame]:
    """Run backtests across multiple indices."""
    
    results = {}
    
    # Fetch macro data once (shared across indices)
    macro_df = fetch_macro_money_supply()
    
    for idx in indices:
        config = INDEX_CONFIG.get(idx)
        if config is None:
            logger.warning(f"Unknown index: {idx}, skipping")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {idx} ({config['name']})")
        logger.info(f"{'='*60}")
        
        try:
            pe_df = fetch_index_pe(config['name'])
            price_df = fetch_index_prices(config['code'])
            
            engine = BacktestEngine(
                macro_df, pe_df, price_df,
                index_name=idx,
                risk_free_rate=risk_free_rate,
                fixed_mix_equity=fixed_mix_equity
            )
            
            results[idx] = run_grid_search(engine, params_grid, periods)
            
        except Exception as e:
            logger.error(f"Failed to process {idx}: {e}")
            continue
    
    return results


# =============================================================================
# EXCEL OUTPUT
# =============================================================================

def export_to_excel(
    results: Dict[str, pd.DataFrame],
    outpath: Path,
    periods: List[int]
) -> None:
    """Export results to Excel with multiple sheets."""
    
    logger.info(f"Exporting results to {outpath}")
    
    with pd.ExcelWriter(outpath, engine='openpyxl') as writer:
        
        overall_best = []
        
        for idx, df in results.items():
            if df.empty:
                continue
            
            # Grid results sheet
            df_sorted = df.sort_values('sharpe_ratio', ascending=False)
            df_sorted.to_excel(writer, sheet_name=f'{idx}_grid', index=False)
            
            # Summary sheet (top 10 per period)
            summary_rows = []
            for period in periods:
                period_df = df[df['period_years'] == period].copy()
                if period_df.empty:
                    continue
                
                top10 = period_df.nlargest(10, 'sharpe_ratio')
                top10['rank'] = range(1, len(top10) + 1)
                summary_rows.append(top10)
                
                # Track best for overall comparison
                best = period_df.loc[period_df['sharpe_ratio'].idxmax()].to_dict()
                best['metric'] = 'best_sharpe'
                overall_best.append(best)
                
                best_excess = period_df.loc[period_df['annualized_excess'].idxmax()].to_dict()
                best_excess['metric'] = 'best_excess'
                overall_best.append(best_excess)
            
            if summary_rows:
                summary_df = pd.concat(summary_rows, ignore_index=True)
                summary_df.to_excel(writer, sheet_name=f'{idx}_summary', index=False)
            
            # Benchmark comparison sheet
            bench_rows = []
            for period in periods:
                period_df = df[df['period_years'] == period]
                if period_df.empty:
                    continue
                
                # Get benchmark metrics from first row (same for all)
                first_row = period_df.iloc[0]
                
                bench_rows.append({
                    'period_years': period,
                    'benchmark': '100% Equity',
                    'ann_return': first_row['benchmark_ann_return'],
                    'volatility': first_row['benchmark_volatility'],
                    'sharpe': first_row['benchmark_sharpe'],
                    'max_dd': first_row['benchmark_max_dd']
                })
                
                bench_rows.append({
                    'period_years': period,
                    'benchmark': f'{int(first_row.get("fixed_mix_equity", 0.6)*100)}/40 Mix',
                    'ann_return': first_row['fixedmix_ann_return'],
                    'sharpe': first_row['fixedmix_sharpe'],
                    'max_dd': first_row['fixedmix_max_dd']
                })
                
                bench_rows.append({
                    'period_years': period,
                    'benchmark': 'Vol-Matched',
                    'ann_return': first_row['volmatch_ann_return'],
                    'sharpe': first_row['volmatch_sharpe']
                })
                
                # Best strategy
                best = period_df.loc[period_df['sharpe_ratio'].idxmax()]
                bench_rows.append({
                    'period_years': period,
                    'benchmark': 'Best Strategy',
                    'ann_return': best['annualized_return'],
                    'volatility': best['volatility'],
                    'sharpe': best['sharpe_ratio'],
                    'max_dd': best['max_drawdown'],
                    'avg_pos': best['avg_position'],
                    'turnover': best['turnover'],
                    'cost_drag': best['cost_drag']
                })
            
            if bench_rows:
                bench_df = pd.DataFrame(bench_rows)
                bench_df.to_excel(writer, sheet_name=f'{idx}_bench', index=False)
        
        # Overall comparison sheet
        if overall_best:
            overall_df = pd.DataFrame(overall_best)
            cols = ['index_name', 'period_years', 'metric', 'sharpe_ratio', 
                    'annualized_return', 'annualized_excess', 'volatility', 
                    'max_drawdown', 'avg_position', 'turnover', 'cost_drag',
                    'base_position', 'trend_sensitivity', 'min_weight', 'max_weight',
                    'cheap_threshold', 'expensive_threshold',
                    'dyn_window_months', 'dyn_quantile', 'pe_window_years', 'band']
            cols = [c for c in cols if c in overall_df.columns]
            overall_df = overall_df[cols]
            overall_df.to_excel(writer, sheet_name='Overall', index=False)
    
    logger.info(f"Excel export complete: {outpath}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest Strategy v2: Multi-Index with Methodology Fixes"
    )
    parser.add_argument(
        '--indices', type=str, nargs='+', default=['CSI300', 'CSI500', 'CSI1000'],
        help='Indices to backtest (default: CSI300 CSI500 CSI1000)'
    )
    parser.add_argument(
        '--periods', type=int, nargs='+', default=[10, 20],
        help='Backtest periods in years (default: 10 20)'
    )
    parser.add_argument(
        '--rf', type=float, default=DEFAULT_RF_ANNUAL,
        help=f'Annual risk-free rate (default: {DEFAULT_RF_ANNUAL})'
    )
    parser.add_argument(
        '--fixed_mix', type=float, default=0.6,
        help='Equity weight for fixed-mix benchmark (default: 0.6)'
    )
    parser.add_argument(
        '--cost_rate', type=float, default=DEFAULT_COST_RATE,
        help=f'Transaction cost rate (default: {DEFAULT_COST_RATE})'
    )
    parser.add_argument(
        '--outdir', type=str, default='./backtest_v2_results',
        help='Output directory (default: ./backtest_v2_results)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Run quick test with minimal parameter grid (~48 combinations)'
    )
    parser.add_argument(
        '--full', action='store_true',
        help='Run full expanded grid search (~216,000 combinations, takes hours)'
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
    
    # Generate parameter grid
    if args.quick:
        # Quick grid: ~48 combinations (for testing)
        params_grid = generate_param_grid(
            dyn_window_months=[48, 60],
            dyn_quantile=[0.20],
            pe_window_years=[7, 10],
            band=[0.0, 0.05],
            base_position=[0.40, 0.50, 0.60],
            trend_sensitivity=[0.30],
            cheap_threshold=[0.20],
            expensive_threshold=[0.80, 0.90],
            max_weight=[1.2],
            min_weight=[0.1, 0.2],
            cost_rate=[args.cost_rate]
        )
        # 2×1×2×2×3×1×1×2×1×2 = 48 combinations
    elif args.full:
        # Full expanded grid: ~40,500 combinations
        params_grid = generate_param_grid(
            dyn_window_months=[24, 36, 48, 60, 72, 84],        # 6
            dyn_quantile=[0.10, 0.15, 0.20, 0.25, 0.30],       # 5
            pe_window_years=[5, 7, 10],                         # 3
            band=[0.0, 0.03, 0.05],                             # 3
            base_position=[0.30, 0.40, 0.50, 0.60, 0.70],       # 5
            trend_sensitivity=[0.20, 0.30],                     # 2
            cheap_threshold=[0.15, 0.20, 0.25],                 # 3
            expensive_threshold=[0.75, 0.80, 0.85, 0.90, 0.95], # 5
            max_weight=[1.2],                                   # 1
            min_weight=[0.0, 0.1, 0.2],                         # 3
            cost_rate=[args.cost_rate]
        )
        # 6×5×3×3×5×2×3×5×1×3 = 40,500 combinations
    else:
        # Default (medium) grid: ~5,400 combinations
        # Reasonable for a few hours run
        params_grid = generate_param_grid(
            dyn_window_months=[36, 48, 60, 72],                 # 4
            dyn_quantile=[0.15, 0.20, 0.25],                    # 3
            pe_window_years=[5, 7, 10],                          # 3
            band=[0.0, 0.03, 0.05],                              # 3
            base_position=[0.30, 0.40, 0.50, 0.60, 0.70],        # 5
            trend_sensitivity=[0.25, 0.30],                      # 2
            cheap_threshold=[0.20],                              # 1
            expensive_threshold=[0.75, 0.80, 0.85, 0.90, 0.95],  # 5
            max_weight=[1.2],                                    # 1
            min_weight=[0.1, 0.2],                               # 2
            cost_rate=[args.cost_rate]
        )
        # 4×3×3×3×5×2×1×5×1×2 = 5,400 combinations
    
    logger.info(f"Generated {len(params_grid)} parameter configurations")
    
    # Run multi-index backtest
    results = run_multi_index_backtest(
        indices=args.indices,
        params_grid=params_grid,
        periods=args.periods,
        risk_free_rate=args.rf,
        fixed_mix_equity=args.fixed_mix
    )
    
    # Export individual CSVs
    for idx, df in results.items():
        if not df.empty:
            df.to_csv(outdir / f'{idx}_results.csv', index=False)
    
    # Export Excel
    excel_path = outdir / 'backtest_results.xlsx'
    export_to_excel(results, excel_path, args.periods)
    
    # Print summary
    print(f"\n{'='*80}")
    print("BACKTEST SUMMARY")
    print(f"{'='*80}")
    
    for idx, df in results.items():
        if df.empty:
            continue
        
        print(f"\n{idx}:")
        for period in args.periods:
            period_df = df[df['period_years'] == period]
            if period_df.empty:
                continue
            
            best = period_df.loc[period_df['sharpe_ratio'].idxmax()]
            print(f"  {period}Y Best Sharpe: {best['sharpe_ratio']:.3f} "
                  f"(Ann: {best['annualized_return']:.2%}, "
                  f"Vol: {best['volatility']:.2%}, "
                  f"MaxDD: {best['max_drawdown']:.2%})")
            print(f"       Params: base_pos={best['base_position']:.2f}, "
                  f"min_wt={best['min_weight']:.2f}, "
                  f"dyn_win={best['dyn_window_months']}m")
            print(f"       vs Benchmark: {best['benchmark_sharpe']:.3f} Sharpe, "
                  f"{best['benchmark_ann_return']:.2%} Ann")
    
    logger.info(f"\nResults saved to {outdir}")
    logger.info(f"Excel file: {excel_path}")


if __name__ == "__main__":
    main()