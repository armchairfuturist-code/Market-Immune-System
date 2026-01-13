"""
Market Immune System - Core Calculation Engine
Detects market fragility through Statistical Turbulence and Absorption Ratio metrics.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.stats import mstats
from sklearn.decomposition import PCA
from pypfopt.risk_models import CovarianceShrinkage
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class SignalStatus(Enum):
    """Market signal status with color codes."""
    GREEN = ("Normal", "#00C853")
    ORANGE = ("Fragile", "#FF9800")
    RED = ("Divergence", "#FF5252")
    BLACK = ("Crash", "#212121")
    BLUE = ("Opportunity", "#2196F3")


@dataclass
class MarketContext:
    """Container for auxiliary market context data."""
    spx_level: float
    spx_50d_ma: float
    vix_level: float
    days_elevated: int
    ai_turbulence: float
    ai_market_ratio: float

@dataclass
class MarketMetrics:
    """Container for market health metrics."""
    turbulence_score: float
    absorption_ratio: float
    signal: SignalStatus
    signal_message: str
    top_contributors: List[Tuple[str, float]]
    spy_return: float
    context: Optional[MarketContext] = None
    hurst_exponent: float = 0.5
    liquidity_z: float = 0.0
    advanced_signal: str = "NORMAL"


class MarketImmuneSystem:
    """
    Core calculation engine for market fragility detection.
    
    Uses Statistical Turbulence (Mahalanobis Distance) and Absorption Ratio (PCA)
    to monitor market health and detect potential crash signals.
    """
    
    # Asset Universe - 99 tickers across 4 groups
    ASSET_UNIVERSE = {
        "Broad Markets": [
            "SPY", "QQQ", "DIA", "IWM", "VXX", "EEM", "EFA", "TLT", "IEF", "SHY",
            "LQD", "HYG", "BND", "AGG", "GLD", "SLV", "CPER", "USO", "UNG", "DBC",
            "PALL", "UUP", "FXE", "FXY", "FXB", "CYB", "XLF", "XLE", "XLK", "XLY",
            "XLI", "XLB", "XLRE", "^VIX", "^VIX3M", "^TNX", "^IRX"
        ],
        "Crypto": [
            "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD",
            "ADA-USD", "AVAX-USD", "DOGE-USD", "DOT-USD", "LINK-USD"
        ],
        "AI & Growth": [
            "NVDA", "AMD", "TSM", "AVGO", "ARM", "MU", "INTC", "TXN", "LRCX", "AMAT",
            "VRT", "ANET", "SMCI", "PLTR", "DELL", "HPE", "CSCO", "IBM", "ORCL", "CEG",
            "VST", "NRG", "CCJ", "URA", "NEE", "SO", "DUK", "TSLA", "PATH", "ISRG",
            "BOTZ", "ROBO", "ARKK"
        ],
        "Defensive": [
            "XLV", "JNJ", "PFE", "MRK", "ABBV", "UNH", "LLY", "AMGN", "XLP", "PG",
            "KO", "PEP", "COST", "WMT", "PM", "XLU", "O", "AMT", "CCI", "PSA",
            "DLR", "USMV", "SPLV"
        ]
    }
    
    def __init__(self, lookback_days: int = 365, fetch_days: int = 1100, min_lookback: int = 60):
        """
        Initialize the Market Immune System.
        
        Args:
            lookback_days: Preferred rolling window for covariance/correlation calculations
            fetch_days: Number of days to fetch from yfinance (buffer for lookback)
            min_lookback: Minimum acceptable lookback window if data is insufficient
        """
        self.lookback_days = lookback_days
        self.fetch_days = fetch_days
        self.min_lookback = min_lookback
        self._effective_lookback = lookback_days  # Will be adjusted after data fetch
        self._all_tickers = self._get_all_tickers()
        
    def _get_all_tickers(self) -> List[str]:
        """Flatten asset universe into a single list of tickers."""
        tickers = []
        for group in self.ASSET_UNIVERSE.values():
            tickers.extend([t.strip() for t in group])
        return tickers
    
    @property
    def effective_lookback(self) -> int:
        """Get the effective lookback window being used."""
        return self._effective_lookback
    
    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fetch price and volume data.
        Returns: (log_returns, prices, volumes)
        """
        try:
            # Download all tickers at once for efficiency
            data = yf.download(
                self._all_tickers,
                period=f"{self.fetch_days}d",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            
            # Extract adjusted close prices
            # Handle MultiIndex columns (Price, Ticker) from yfinance
            if isinstance(data.columns, pd.MultiIndex):
                if 'Close' in data.columns.get_level_values(0):
                    prices = data['Close']
                else:
                    # Fallback or different structure
                    prices = data
            elif 'Close' in data.columns:
                prices = data['Close']
            else:
                prices = data
            
            # Forward fill missing data (crucial for crypto/stock alignment)
            prices = prices.ffill()
            
            # Drop columns with >10% missing data
            missing_pct = prices.isnull().sum() / len(prices)
            valid_cols = missing_pct[missing_pct <= 0.10].index
            prices = prices[valid_cols]
            
            # Drop any remaining NaN rows
            prices = prices.dropna()
            
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1))
            
            # Clean infinite values and NaNs
            log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna()
            
            # Filter out days with >90% zero returns (holidays/weekends artifacts)
            non_zero_pct = (log_returns != 0).mean(axis=1)
            log_returns = log_returns[non_zero_pct > 0.1]
            
            # Align prices and volume to clean returns index
            common_idx = log_returns.index
            prices = prices.reindex(common_idx)
            
            # Handle Volume
            # yfinance returns volume as part of data if not multi-index, or MultiIndex
            # We need to extract it similar to Close
            if isinstance(data.columns, pd.MultiIndex):
                if 'Volume' in data.columns.get_level_values(0):
                    volume = data['Volume']
                else:
                    volume = pd.DataFrame(1, index=data.index, columns=prices.columns) # Fallback
            elif 'Volume' in data.columns:
                 volume = data['Volume']
            else:
                 volume = pd.DataFrame(1, index=data.index, columns=prices.columns)

            volume = volume.ffill().reindex(common_idx)
            
            # Dynamically adjust lookback if insufficient data
            available_days = len(log_returns)
            if available_days < self.lookback_days:
                adjusted_lookback = max(self.min_lookback, int(available_days * 0.6))
                self._effective_lookback = adjusted_lookback
            else:
                 self._effective_lookback = self.lookback_days
                 
            return log_returns, prices, volume
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Return empty DFs on failure
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def calculate_turbulence(self, returns: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> Tuple[float, pd.Series]:
        """
        Calculate Statistical Turbulence using Mahalanobis Distance.
        
        Args:
            returns: DataFrame of log returns
            target_date: Date to calculate turbulence for (default: latest)
            
        Returns:
            Tuple of (turbulence_score, contribution_series)
        """
        if target_date is None:
            target_date = returns.index[-1]
        
        # Get the lookback window (use effective lookback)
        date_idx = returns.index.get_loc(target_date)
        lookback = self._effective_lookback
        
        if date_idx < lookback:
            # If still insufficient, use what we have (but at least min_lookback)
            lookback = max(self.min_lookback, date_idx)
        
        lookback_returns = returns.iloc[date_idx - lookback:date_idx]
        current_return = returns.loc[target_date].values
        
        # 1. Winsorization: Clip extreme outliers at 2.5% and 97.5%
        # This stops a single bad 'yfinance' print from breaking the system
        try:
            # winsorize along axis 0 (columns/assets)
            clipped_returns = mstats.winsorize(lookback_returns.values, limits=[0.025, 0.025], axis=0)
            clipped_df = pd.DataFrame(clipped_returns, columns=lookback_returns.columns)
        except:
            clipped_df = lookback_returns

        # 2. Calculate mean return vector from clipped data
        mu = clipped_df.mean().values
        
        # 3. Covariance with Shrinkage & Regularization
        try:
            # Using Ledoit-Wolf Shrinkage as base (PRD Requirement)
            shrinkage = CovarianceShrinkage(clipped_df)
            cov_matrix = shrinkage.ledoit_wolf()
            
            # Add a tiny bit of noise to the diagonal to stabilize (Tikhonov regularization)
            epsilon = 1e-4
            cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
            
            # 4. Use Pseudo-Inverse instead of standard Inverse for extreme stability
            cov_inv = np.linalg.pinv(cov_matrix)
                
        except Exception:
            # Ultimate fallback
            cov_matrix = clipped_df.cov().values
            epsilon = 1e-4
            cov_matrix += np.eye(cov_matrix.shape[0]) * epsilon
            cov_inv = np.linalg.pinv(cov_matrix)
        
        # 5. Calculate Mahalanobis distance squared
        diff = current_return - mu
        mahal_sq = np.dot(np.dot(diff, cov_inv), diff)
        
        # Calculate contribution per asset (partial Mahalanobis)
        # Using the stabilized inverse
        contributions = pd.Series(index=returns.columns, dtype=float)
        for i, col in enumerate(returns.columns):
            partial_diff = np.zeros_like(diff)
            partial_diff[i] = diff[i]
            partial_mahal = np.dot(np.dot(partial_diff, cov_inv), partial_diff)
            contributions[col] = partial_mahal
        
        # Normalize contributions to sum to total
        if contributions.sum() > 0:
            contributions = contributions / contributions.sum() * mahal_sq
        
        return float(mahal_sq), contributions
    
    def calculate_rolling_turbulence(self, returns: pd.DataFrame, ema_span: int = 3) -> pd.Series:
        """
        Calculate rolling turbulence scores with EMA smoothing.
        Uses an Expanding Window strategy for the start period to maximize visibility.
        """
        turbulence_scores = []
        # Start calculating as soon as we have min_lookback days
        start_idx = self.min_lookback
        valid_dates = returns.index[start_idx:]
        
        for date in valid_dates:
            try:
                # Dynamic Lookback: Expand from min_lookback up to effective_lookback
                date_idx = returns.index.get_loc(date)
                
                # We want to use as much history as available, up to the limit
                available_history = date_idx
                current_lookback = min(available_history, self._effective_lookback)
                
                # Check if we meet the minimum requirement
                if current_lookback < self.min_lookback:
                    turbulence_scores.append(np.nan)
                    continue
                
                # Slice logic is handled inside calculate_turbulence, but we need to ensure
                # we don't accidentally pass a date that forces a crash.
                # Actually, calculate_turbulence already handles "use effective lookback".
                # We just need to trick it or rely on its internal clipping?
                # calculate_turbulence logic: "if date_idx < lookback: lookback = max(min_lookback, date_idx)"
                # This ALREADY implements the expanding window logic internally!
                # The issue was the loop range in THIS function.
                
                score, _ = self.calculate_turbulence(returns, date)
                turbulence_scores.append(score)
            except Exception:
                turbulence_scores.append(np.nan)
        
        scores_series = pd.Series(turbulence_scores, index=valid_dates)
        
        # Apply EMA smoothing
        smoothed = scores_series.ewm(span=ema_span, adjust=False).mean()
        
        return smoothed
    
    def calculate_absorption_ratio(self, returns: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> float:
        """
        Calculate Absorption Ratio using PCA.
        
        Measures how much variance is explained by the top 20% of principal components,
        indicating market unification/fragility.
        
        Args:
            returns: DataFrame of log returns
            target_date: Date to calculate absorption for (default: latest)
            
        Returns:
            Absorption ratio score (0-1000)
        """
        if target_date is None:
            target_date = returns.index[-1]
        
        # Get the lookback window (use effective lookback)
        date_idx = returns.index.get_loc(target_date)
        lookback = self._effective_lookback
        
        if date_idx < lookback:
            lookback = max(self.min_lookback, date_idx)
        
        lookback_returns = returns.iloc[date_idx - lookback:date_idx]
        
        # Calculate correlation matrix
        corr_matrix = lookback_returns.corr()
        
        # Handle NaNs in correlation matrix (e.g., constant assets)
        corr_matrix = corr_matrix.fillna(0)
        
        # Fit PCA
        pca = PCA()
        pca.fit(corr_matrix)
        
        # Calculate variance explained by top 20% of components
        n_components = len(corr_matrix.columns)
        top_n = max(1, int(n_components * 0.20))
        
        top_variance = pca.explained_variance_ratio_[:top_n].sum()
        total_variance = pca.explained_variance_ratio_.sum()
        
        # Scale to 0-1000
        absorption_score = (top_variance / total_variance) * 1000
        
        return float(absorption_score)
    
    def calculate_rolling_absorption(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate rolling absorption ratio.
        
        Args:
            returns: DataFrame of log returns
            
        Returns:
            Series of absorption ratio scores
        """
        absorption_scores = []
        valid_dates = returns.index[self._effective_lookback:]
        
        for date in valid_dates:
            try:
                score = self.calculate_absorption_ratio(returns, date)
                absorption_scores.append(score)
            except Exception:
                absorption_scores.append(np.nan)
        
        return pd.Series(absorption_scores, index=valid_dates)
    
    def calculate_rolling_correlation(self, returns: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix for the heatmap.
        
        Args:
            returns: DataFrame of log returns
            window: Rolling window in days
            
        Returns:
            Correlation matrix for the last `window` days
        """
        recent_returns = returns.tail(window)
        return recent_returns.corr()
    
    def calibrate_turbulence_score(self, raw_score: float, history_series: pd.Series) -> float:
        """
        Calibrate raw Mahalanobis distance to a 0-1000 scale.
        Anchor: 99th Percentile of history = 370.
        Cap: 1000.
        """
        if len(history_series) < 100:
            # Fallback for insufficient history
            return min(raw_score, 1000.0)
            
        p99 = history_series.quantile(0.99)
        if p99 == 0: return 0.0
        
        # Scaling formula: (Raw / P99) * 370
        scaled_score = (raw_score / p99) * 370
        return min(scaled_score, 1000.0)

    def generate_signal(
        self,
        spx_return: float,
        turbulence_score: float,
        absorption_score: float,
        n_assets: int,
        prev_turbulence: Optional[float] = None
    ) -> Tuple[SignalStatus, str]:
        """
        Generate market signal based on Calibrated Metrics (0-1000 Scale).
        Thresholds: Warning > 180, Critical > 370.
        """
        # Thresholds defined in PRD Section 3
        THRESHOLD_WARNING = 180.0
        THRESHOLD_CRITICAL = 370.0
        
        # Check for opportunity (mean reversion)
        if prev_turbulence is not None:
            if prev_turbulence > THRESHOLD_CRITICAL and turbulence_score < 300:
                return SignalStatus.BLUE, "OPPORTUNITY: Mean reversion detected. Consider buying."
        
        # Check for crash (Extreme Turbulence + Drop)
        if spx_return < -1.0 and turbulence_score > THRESHOLD_CRITICAL:
            return SignalStatus.BLACK, "CRASH: Severe drawdown with extreme turbulence."
        
        # Check for divergence (Rising Market + Warning Turbulence)
        if spx_return > 0 and turbulence_score > THRESHOLD_WARNING:
            return SignalStatus.RED, "DIVERGENCE: Market rising on broken structure."
        
        # Check for fragility (Absorption)
        if absorption_score > 800:
            return SignalStatus.ORANGE, "FRAGILE: High market unification. Risk of cascade."
        
        # Check for elevated turbulence
        if turbulence_score > THRESHOLD_WARNING:
            return SignalStatus.ORANGE, f"ELEVATED: Turbulence above warning level ({THRESHOLD_WARNING})."
        
        # Normal conditions
        return SignalStatus.GREEN, "NORMAL: Market structure intact."
    
    def _estimate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Helper to estimate covariance matrix using Ledoit-Wolf shrinkage."""
        try:
            shrinkage = CovarianceShrinkage(returns)
            cov_matrix = shrinkage.ledoit_wolf()
        except Exception:
            cov_matrix = returns.cov()
        return cov_matrix

    def calculate_sector_turbulence(self, returns: pd.DataFrame, sector_name: str = "AI & Growth") -> float:
        """Calculate turbulence score specifically for a sector subset."""
        if sector_name not in self.ASSET_UNIVERSE:
            return 0.0
            
        sector_assets = [t for t in self.ASSET_UNIVERSE[sector_name] if t in returns.columns]
        if len(sector_assets) < 2:
            return 0.0
            
        sector_returns = returns[sector_assets]
        
        # Calculate covariance for sector
        try:
            cov_matrix = self._estimate_covariance(sector_returns)
            # Inverse covariance
            inv_cov = np.linalg.pinv(cov_matrix.values) if np.linalg.det(cov_matrix.values) == 0 else np.linalg.inv(cov_matrix.values)
            
            # Mahalanobis for latest day
            latest = sector_returns.iloc[-1].values
            mean = sector_returns.mean().values
            diff = latest - mean
            
            dist_sq = diff.dot(inv_cov).dot(diff)
            return dist_sq
        except:
             return 0.0

    def get_market_cycle_status(self, returns: pd.DataFrame) -> Dict:
        """
        Determines the current Economic Cycle Phase based on Sector Rotation.
        Logic: Compare 3-month Relative Strength (RS) of key sectors vs SPY.
        """
        # Define Cycle Proxies (Your Cheat Sheet)
        cycles = {
            "Early Cycle": ["XLF", "XLY", "IWM"],     # Financials, Discretionary, Small Caps
            "Mid Cycle":   ["XLK", "XLI"],            # Tech, Industrials
            "Late Cycle":  ["XLE", "XLB", "XLP"],     # Energy, Materials, Staples
            "Recession":   ["XLU", "XLV", "SHY"]      # Utilities, Healthcare, Cash/Bonds
        }
        
        # Calculate 60-day (approx 3-month) cumulative return for all assets
        # Ensure we look at the END of the data
        if len(returns) < 60:
             return {}
             
        recent_ret = np.exp(returns.tail(60).cumsum().iloc[-1]) - 1
        
        if "SPY" not in recent_ret:
            return {}
            
        spy_perf = recent_ret["SPY"]
        
        cycle_scores = {}
        
        for phase, tickers in cycles.items():
            # Filter for tickers we actually have data for
            valid = [t for t in tickers if t in recent_ret]
            if not valid:
                continue
                
            # Calculate Average Relative Performance vs SPY
            # Positive = Outperforming SPY
            rel_perf = [(recent_ret[t] - spy_perf) for t in valid]
            avg_outperformance = np.mean(rel_perf) * 100 # percentage
            
            cycle_scores[phase] = avg_outperformance
            
        if not cycle_scores:
            return {}
            
        # Find the winning cycle
        current_phase = max(cycle_scores, key=cycle_scores.get)
        
        return {
            "current_phase": current_phase,
            "strength": cycle_scores[current_phase],
            "details": cycle_scores
        }

    @staticmethod
    def get_hurst_exponent(time_series: np.array, max_lag: int = 20) -> float:
        """
        Calculates the Hurst Exponent (H) to detect market fragility.
        H > 0.75 signalizes a 'crowded trade' prone to sharp reversal.
        Using Log Prices as time_series input.
        """
        try:
            lags = range(2, max_lag)
            # Tau = StdDev of (Price(t+lag) - Price(t)) => StdDev of Log Returns over 'lag'
            tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
            
            # Polyfit log(tau) vs log(lag)
            # Slope m = H
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = m[0]
            # Standard Hurst ranges 0-1. 
            # If time_series is geometric brownian motion, variance scales with t. Std scales with t^0.5. slope=0.5.
            return hurst
        except:
            return 0.5

    def check_liquidity_stress(self, returns: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, ticker: str = "SPY") -> float:
        """
        Calculates Amihud Illiquidity Proxy Z-Score for a specific ticker (SPY default).
        """
        if ticker not in returns.columns or ticker not in volumes.columns or ticker not in prices.columns:
            return 0.0
            
        # Aligned subset
        r = returns[ticker].abs()
        p = prices[ticker]
        v = volumes[ticker]
        
        # Dollar Volume
        dollar_vol = p * v
        dollar_vol = dollar_vol.replace(0, np.nan).ffill()
        
        # Illiq
        illiq = r / dollar_vol
        
        # Smooth
        illiq_smooth = illiq.rolling(window=10).mean()
        
        # Normalize (Z-Score vs 365d history)
        history = illiq_smooth.iloc[-365:]
        if len(history) < 20: return 0.0
        
        mu = history.mean()
        sigma = history.std()
        current = illiq_smooth.iloc[-1]
        
        if sigma == 0: return 0.0
        
        return (current - mu) / sigma

    def calculate_rolling_liquidity(self, returns: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
        """
        Calculate rolling Amihud Illiquidity Z-Score efficiently (Vectorized).
        """
        if ticker not in returns.columns or ticker not in volumes.columns or ticker not in prices.columns:
            return pd.Series(0.0, index=returns.index)
            
        # Aligned subset
        r = returns[ticker].abs()
        p = prices[ticker]
        v = volumes[ticker]
        
        # Dollar Volume
        dollar_vol = p * v
        dollar_vol = dollar_vol.replace(0, np.nan).ffill()
        
        # Illiq (Daily)
        illiq = r / dollar_vol
        
        # Smooth (10-day MA)
        illiq_smooth = illiq.rolling(window=10).mean()
        
        # Rolling Statistics (365-day Window) for Z-Score
        # We need the mean and std of the *smoothed* series over the *past* 365 days
        rolling_mu = illiq_smooth.rolling(window=365).mean()
        rolling_sigma = illiq_smooth.rolling(window=365).std()
        
        # Z-Score
        z_score = (illiq_smooth - rolling_mu) / rolling_sigma
        
        return z_score.fillna(0.0)

    def get_futures_sentiment(self) -> Dict:
        """
        Compares Futures vs Spot to detect Backwardation (Bearish) or Contango (Bullish).
        """
        try:
            # Fetch Spot and Futures
            tickers = ["^GSPC", "ES=F", "BTC-USD", "BTC=F"]
            data = yf.download(tickers, period="5d", progress=False)["Close"].iloc[-1]
            
            # 1. S&P 500 Basis
            spx_spot = data.get("^GSPC", 0)
            spx_fut = data.get("ES=F", 0)
            
            spx_basis = 0.0
            spx_signal = "N/A"
            
            if spx_spot > 0 and spx_fut > 0:
                spx_basis = ((spx_fut - spx_spot) / spx_spot) * 100
                spx_signal = "BEARISH (Backwardation)" if spx_basis < -0.02 else "NORMAL (Contango)"
            
            # 2. Bitcoin Basis
            btc_spot = data.get("BTC-USD", 0)
            btc_fut = data.get("BTC=F", 0)
            
            btc_basis = 0.0
            btc_signal = "N/A"
            
            if btc_spot > 0 and btc_fut > 0:
                btc_basis = ((btc_fut - btc_spot) / btc_spot) * 100
                btc_signal = "INSTITUTIONAL SHORTING" if btc_basis < -0.5 else "NORMAL"
            
            return {
                "spx_basis": spx_basis,
                "spx_signal": spx_signal,
                "btc_basis": btc_basis,
                "btc_signal": btc_signal
            }
        except Exception as e:
            print(f"Futures Error: {e}")
            return {}

    def get_vix_term_structure_signal(self, prices: pd.DataFrame) -> str:
        """Check Backwardation (Spot > 3M). Uses VIX3M or VXV."""
        if '^VIX' not in prices.columns:
            return "N/A (VIX Missing)"
            
        # Try finding 3-month VIX ticker
        term_ticker = None
        if '^VIX3M' in prices.columns:
            term_ticker = '^VIX3M'
        elif '^VXV' in prices.columns:
            term_ticker = '^VXV'
            
        if term_ticker:
            spot = prices['^VIX'].iloc[-1]
            term = prices[term_ticker].iloc[-1]
            
            if term == 0: return "N/A (Zero Term)"
            
            ratio = spot / term
            if ratio > 1.0:
                return f"BACKWARDATION (Panic) {ratio:.2f}"
            return f"Contango (Normal) {ratio:.2f}"
            
        return "N/A (3M VIX Missing)"

    def get_advanced_signal(self, turbulence: float, liquidity_z: float, hurst: float, absorption: float = 0.0) -> str:
        """The Super-Signal logic. Downgrades BUY signals if Absorption is high."""
        
        # 1. Fragility Check (Absorption Override)
        if absorption > 850:
            if turbulence < 300 and liquidity_z < 1.0:
                return "Caution: Fragile Buy (High Absorption)"
            return "Warning: Market Locked in Lockstep"

        # 2. Standard Logic
        if hurst > 0.75 and liquidity_z > 2.0:
            return "Danger: Liquidity Hole Forming"
        
        if turbulence > 370: # Calibrated Critical Threshold
            return "Critical: Structural Break Detected"
            
        if turbulence < 180 and liquidity_z < 1.0 and hurst < 0.5:
            return "Strong Buy: Liquidity Restored"
            
        return "Normal Market Conditions"

    def get_crash_forecast(self, turbulence: pd.Series, prices: pd.DataFrame, ticker: str = "^GSPC") -> Dict:
        """
        Estimate crash probability based on historical Divergence signals.
        Returns: {probability, avg_lead_time, sample_size}
        """
        default_res = {"probability": 0.0, "avg_lead_time": 0.0, "sample_size": 0}
        
        if ticker not in prices.columns:
            # Fallback to SPY if GSPC missing
            if "SPY" in prices.columns:
                ticker = "SPY"
            else:
                return default_res
            
        p = prices[ticker]
        
        # Align series
        common = turbulence.index.intersection(p.index)
        if len(common) < 100:
             return default_res
             
        t = turbulence.loc[common]
        p_aligned = p.loc[common]
        
        # Calculate SMA50
        sma = p_aligned.rolling(50).mean()
        
        # Divergence: High Turb (>95th percentile) AND Rising Price (Price > SMA)
        thresh = t.quantile(0.95)
        rising = p_aligned > sma
        
        signals = (t > thresh) & rising
        
        # Analyze forward returns for signal dates
        crash_count = 0
        total_signals = 0
        lead_times = []
        
        # Group signals by week (approx) to avoid counting same event multiple times
        i = 0
        dates = signals.index
        while i < len(dates) - 20:
            if signals.iloc[i]:
                total_signals += 1
                # Check next 20 days for drawdown > 3%
                start_price = p_aligned.iloc[i]
                future_prices = p_aligned.iloc[i+1 : i+21]
                if len(future_prices) > 0:
                    min_price = future_prices.min()
                    drawdown = (min_price - start_price) / start_price
                    
                    if drawdown < -0.03:
                        crash_count += 1
                        # Estimate lead time: Day of min price
                        days_to_min = (future_prices.idxmin() - dates[i]).days
                        lead_times.append(days_to_min)
                
                i += 20 # Skip forward to avoid overlapping signals
            else:
                i += 1
                
        prob = (crash_count / total_signals * 100) if total_signals > 0 else 0.0
        avg_lead = np.mean(lead_times) if lead_times else 0.0
        
        return {
            "probability": float(prob),
            "avg_lead_time": float(avg_lead),
            "sample_size": int(total_signals)
        }

    def get_detailed_report(self, metrics: MarketMetrics) -> Dict[str, str]:
        """
        Generate a detailed explanation of the current state, 
        defining what 'Healthy' means vs current reality.
        """
        
        # 1. State Verification
        turb_norm = stats.chi2.cdf(metrics.turbulence_score, 99) * 1000
        
        state = "UNKNOWN"
        reason = ""
        healthy_target = "Turbulence < 750 (75th Percentile)"
        
        if turb_norm < 750:
            state = "HEALTHY"
            reason = f"Turbulence ({turb_norm:.0f}) is within normal noise levels (Bottom 75%). No structural stress detected."
            action = "Maintain optimal exposure. Market functioning normally."
        elif turb_norm < 950:
            state = "AT RISK (ELEVATED)"
            reason = f"Turbulence ({turb_norm:.0f}) is Elevated (Top 25%). Volatility clustering detected."
            action = "Monitor closely. Tighten stops. Avoid aggressive leverage."
        else:
            state = "CRITICAL"
            reason = f"Turbulence ({turb_norm:.0f}) is Extreme (Top 5%). Values > 950 indicate a 2-Sigma structural break."
            action = "Reduce Risk. Hedge Tails. Market is fragile and prone to cascade."

        # 2. Divergence Logic
        # VIX is explicitly passed via Context usually, but we can infer or use general logic
        # If State is Critical but VIX is low (we don't have VIX here usually, only in Context)
        # We'll use the Advanced Signal "Fragile" text if available
        if "FRAGILE" in metrics.advanced_signal:
             reason += " **Divergence Confirmed**: Hurst Exponent (>0.75) shows crowding despite calm price action."

        return {
            "current_state": state,
            "verification_math": reason,
            "definition_of_healthy": healthy_target,
            "recommended_action": action,
            "thresholds": "Healthy: 0-750 | Elevated: 750-950 | Critical: 950-1000"
        }

    def get_current_metrics(self, returns: pd.DataFrame, prices: pd.DataFrame = None, volumes: pd.DataFrame = None, target_date: pd.Timestamp = None) -> MarketMetrics:
        """
        Calculate all current market metrics.
        
        Args:
            returns: DataFrame of log returns
            prices: DataFrame of prices (optional, for advanced metrics)
            volumes: DataFrame of volumes (optional, for advanced metrics)
            target_date: Date to calculate metrics for (default: latest)
            
        Returns:
            MarketMetrics containing all current values
        """
        if target_date is None:
            target_date = returns.index[-1]
            
        # Get turbulence and contributions
        turbulence_raw, contributions = self.calculate_turbulence(returns, target_date)
        
        # Get previous day turbulence for signal
        prev_turbulence = None
        # Find index of target date
        if target_date in returns.index:
            loc = returns.index.get_loc(target_date)
            if loc > 0:
                prev_date = returns.index[loc-1]
                try:
                    prev_turbulence, _ = self.calculate_turbulence(returns, prev_date)
                except Exception:
                    pass
        
        # Get absorption ratio
        absorption = self.calculate_absorption_ratio(returns, target_date)
        
        # Get SPY return
        spy_return = 0.0
        if 'SPY' in returns.columns and target_date in returns.index:
            spy_return = returns.loc[target_date, 'SPY'] * 100  # Convert to percentage
        
        # Generate primary signal
        signal, message = self.generate_signal(
            spy_return, turbulence_raw, absorption, len(returns.columns), prev_turbulence
        )
        
        # Get top 5 contributors
        top_contributors = contributions.nlargest(5)
        top_contributors_list = [
            (ticker, float(value)) for ticker, value in top_contributors.items()
        ]
        
        # Advanced Metrics
        hurst = 0.5
        liquidity_z = 0.0
        adv_signal = "NORMAL"
        
        if prices is not None and volumes is not None:
            # Hurst (on SPY log prices) - SLICED TO 300 DAYS ending at target_date
            if "SPY" in prices.columns:
                 # Ensure we only use data up to target_date
                 hist_prices = prices.loc[:target_date, "SPY"]
                 log_p = np.log(hist_prices.iloc[-300:].values) # Fix for Hurst Anomaly
                 hurst = self.get_hurst_exponent(log_p)
            
            # Liquidity - Calculate for target_date
            # check_liquidity_stress calculates for the *latest* in the passed DF
            # So we pass sliced data
            sliced_returns = returns.loc[:target_date]
            sliced_prices = prices.loc[:target_date]
            sliced_volumes = volumes.loc[:target_date]
            liquidity_z = self.check_liquidity_stress(sliced_returns, sliced_prices, sliced_volumes, "SPY")
            
            # Advanced Signal (Pass Absorption!)
            # Calibrate turbulence roughly for the signal logic (P99 ~ 124 for df=90)
            df_assets = len(returns.columns)
            p99_est = stats.chi2.ppf(0.99, df=df_assets)
            turb_calibrated = (turbulence_raw / p99_est) * 370
            
            adv_signal = self.get_advanced_signal(turb_calibrated, liquidity_z, hurst, absorption)

        return MarketMetrics(
            turbulence_score=turbulence_raw, # Return RAW here, main.py calibrates it properly
            absorption_ratio=absorption,
            signal=signal,
            signal_message=message,
            top_contributors=top_contributors_list,
            spy_return=spy_return,
            hurst_exponent=hurst,
            liquidity_z=liquidity_z,
            advanced_signal=adv_signal
        )
    
    def get_spy_cumulative_returns(self, returns: pd.DataFrame) -> pd.Series:
        """
        Calculate cumulative returns for SPY.
        
        Args:
            returns: DataFrame of log returns
            
        Returns:
            Series of cumulative returns (as percentages)
        """
        if 'SPY' not in returns.columns:
            raise ValueError("SPY not found in returns data")
        
        spy_returns = returns['SPY']
        cumulative = (np.exp(spy_returns.cumsum()) - 1) * 100
        
        return cumulative

    def fetch_market_context_data(self) -> Tuple[float, float, float]:
        """
        Fetch auxiliary market data (SPX Level, SPX 50d MA, VIX Level).
        Returns: (spx_price, spx_50d_ma, vix_price)
        """
        try:
            # Fetch SPX and VIX
            aux_data = yf.download(["^GSPC", "^VIX"], period="6mo", progress=False, threads=True)
            
            # Extract SPX and VIX safely
            # yfinance structure varies (Price, Ticker) or just (Ticker)
            if isinstance(aux_data.columns, pd.MultiIndex):
                try:
                    spx = aux_data["Close"]["^GSPC"].dropna()
                    vix = aux_data["Close"]["^VIX"].dropna()
                except KeyError:
                    # Try accessing level 1 directly if level 0 is not named 'Close'
                    spx = aux_data.xs('^GSPC', level=1, axis=1)["Close"].dropna()
                    vix = aux_data.xs('^VIX', level=1, axis=1)["Close"].dropna()
            else:
                spx = aux_data["Close"] if "^GSPC" not in aux_data.columns else aux_data["^GSPC"]
                vix = aux_data["Close"] if "^VIX" not in aux_data.columns else aux_data["^VIX"]

            current_spx = float(spx.iloc[-1])
            spx_ma = float(spx.rolling(window=50).mean().iloc[-1])
            current_vix = float(vix.iloc[-1])
            
            return current_spx, spx_ma, current_vix
        except Exception as e:
            print(f"Error fetching context data: {e}")
            return 0.0, 0.0, 0.0

    def calculate_days_elevated(self, turbulence_series: pd.Series, threshold: float) -> int:
        """
        Calculate consecutive days the turbulence has been above the threshold.
        """
        if len(turbulence_series) == 0:
            return 0
            
        # If today is NOT elevated, the streak is 0
        if turbulence_series.iloc[-1] <= threshold:
            return 0
            
        count = 0
        # Iterate backwards from the latest date
        for score in reversed(turbulence_series.values):
            if score > threshold:
                count += 1
            else:
                break
        return count

    def get_turbulence_drivers(self, returns: pd.DataFrame, top_n: int = 5) -> list:
        """
        Identifies which assets are driving the high turbulence score using absolute Z-scores.
        """
        if len(returns) < 2:
            return []
            
        # Latest Returns
        latest_ret = returns.iloc[-1]
        
        # Deviation scaled by Volatility (Full History Mean/Std)
        mu = returns.mean()
        sigma = returns.std().replace(0, np.nan)
        
        z_scores = (latest_ret - mu) / sigma
        
        # Create DataFrame for sorting
        scores = pd.DataFrame({
            'Ticker': z_scores.index,
            'Z_Score': z_scores.values,
            'Return': latest_ret.values,
            'Abs_Z': np.abs(z_scores.values)
        }).sort_values('Abs_Z', ascending=False)
        
        # Format results for the Streamlit UI
        drivers = []
        for _, row in scores.head(top_n).iterrows():
            drivers.append({
                'ticker': row['Ticker'],
                'z_score': row['Z_Score'],
                'return': row['Return'] * 100 # Convert to %
            })
            
        return drivers

    def get_macro_signals(self, returns: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> List[Dict]:
        """
        Analyze macro-economic ratios for institutional recommendations.
        Uses cumulative returns to approximate price trends relative to the specific target_date.
        """
        signals = []
        
        # 1. Handle Target Date
        if target_date is None:
            target_date = returns.index[-1]
            
        # 2. Slice Data to the Target Date (No looking into the future!)
        # We need enough history for cumulative calc, so we take everything UP TO target_date
        valid_returns = returns.loc[:target_date]
        
        if len(valid_returns) < 20:
            return signals
            
        # 3. Reconstruct cumulative returns
        cum_ret = np.exp(valid_returns.cumsum())
        
        # Helper to check trend (uses the END of the sliced series)
        def check_trend(series, name, bullish_msg, bearish_msg, url, description):
            # Simple 20-day trend leading up to target_date
            recent = series.tail(20)
            
            # Ensure we actually have data for this specific date
            if len(recent) < 20: return None
            
            # Slope of normalized line
            y = recent.values
            x = np.arange(len(y))
            slope, _, _, _, _ = stats.linregress(x, y)
            
            trend_strength = slope * 100 
            
            if trend_strength > 0.05:
                return {
                    "pair": name, 
                    "trend": "Rising", 
                    "signal": bullish_msg, 
                    "strength": trend_strength,
                    "url": url,
                    "desc": description
                }
            elif trend_strength < -0.05:
                return {
                    "pair": name, 
                    "trend": "Falling", 
                    "signal": bearish_msg, 
                    "strength": trend_strength,
                    "url": url,
                    "desc": description
                }
            return None

        # Check GLD Trend first for Context
        gld_rising = False
        if 'GLD' in cum_ret.columns and 'SPY' in cum_ret.columns:
            gld_ratio = cum_ret['GLD'] / cum_ret['SPY']
            # Quick slope check
            recent_gld = gld_ratio.iloc[-20:]
            slope_gld, _, _, _, _ = stats.linregress(np.arange(len(recent_gld)), recent_gld.values)
            if slope_gld > 0: gld_rising = True

        # 1. EM vs US (EEM / SPY)
        if 'EEM' in cum_ret.columns and 'SPY' in cum_ret.columns:
            ratio = cum_ret['EEM'] / cum_ret['SPY']
            sig = check_trend(
                ratio, "EEM/SPY", 
                "Overweight Emerging Markets",
                "US Exceptionalism Dominating",
                "https://www.investopedia.com/articles/investing/092815/emerging-markets-vs-developed-markets.asp",
                "Compares strength of Emerging Markets vs S&P 500."
            )
            if sig: signals.append(sig)

        # 2. Risk Preference (SPY / TLT)
        if 'SPY' in cum_ret.columns and 'TLT' in cum_ret.columns:
            ratio = cum_ret['SPY'] / cum_ret['TLT']
            
            # Contextualize "Risk-On"
            bull_msg = "Risk-On: Stocks > Bonds"
            if gld_rising:
                bull_msg = "Reflationary Melt-Up (Debasement)"
            
            sig = check_trend(
                ratio, "SPY/TLT", 
                bull_msg,
                "Risk-Off: Flight to Bonds",
                "https://stockcharts.com/school/doku.php?id=chart_school:market_analysis:intermarket_analysis",
                "The classic Risk-On/Risk-Off gauge. If Gold is also rising, this indicates currency debasement, not just growth."
            )
            if sig: signals.append(sig)

        # 3. Small Cap Strength (IWM / SPY)
        if 'IWM' in cum_ret.columns and 'SPY' in cum_ret.columns:
            ratio = cum_ret['IWM'] / cum_ret['SPY']
            sig = check_trend(
                ratio, "IWM/SPY", 
                "Broadening Rally (Bullish)",
                "Narrow Rally (Mega-Cap Focus)",
                "https://www.investopedia.com/terms/b/breadthofmarket.asp",
                "Small Caps vs Large Caps. A healthy bull market requires participation from small companies (Rising)."
            )
            if sig: signals.append(sig)
            
        # 4. Safe Haven (GLD / SPY)
        if 'GLD' in cum_ret.columns and 'SPY' in cum_ret.columns:
            ratio = cum_ret['GLD'] / cum_ret['SPY']
            sig = check_trend(
                ratio, "GLD/SPY", 
                "Defensive Rotation: Gold Leading",
                "Growth Focus: Gold Lagging",
                "https://www.gold.org/goldhub/research/gold-as-a-strategic-asset",
                "Gold vs Stocks. Rising indicates fear or inflation hedging dominating equity growth."
            )
            if sig: signals.append(sig)

        # 5. Economic Cycle (XLY / XLP)
        if 'XLY' in cum_ret.columns and 'XLP' in cum_ret.columns:
            ratio = cum_ret['XLY'] / cum_ret['XLP']
            sig = check_trend(
                ratio, "XLY/XLP", 
                "Confident Consumer (Cyclical)",
                "Defensive Posturing (Staples)",
                "https://school.stockcharts.com/doku.php?id=market_analysis:sector_rotation_analysis",
                "Discretionary vs Staples. Rising suggests economic confidence; Falling suggests recessionary fears."
            )
            if sig: signals.append(sig)

        # 6. Credit Stress (HYG / LQD)
        if 'HYG' in cum_ret.columns and 'LQD' in cum_ret.columns:
            ratio = cum_ret['HYG'] / cum_ret['LQD']
            sig = check_trend(
                ratio, "HYG/LQD", 
                "Credit Appetite (Junk Outperforming)",
                "Credit Stress (Quality Flight)",
                "https://www.investopedia.com/terms/h/high_yield_bond.asp",
                "High Yield vs Investment Grade. Falling ratio indicates credit stress and widening spreads."
            )
            if sig: signals.append(sig)

        # 7. Growth vs Stagflation (CPER / GLD)
        if 'CPER' in cum_ret.columns and 'GLD' in cum_ret.columns:
            ratio = cum_ret['CPER'] / cum_ret['GLD']
            sig = check_trend(
                ratio, "CPER/GLD", 
                "Reflationary Growth (Dr. Copper)",
                "Stagflation Risk (Gold Safety)",
                "https://www.cmegroup.com/education/featured-reports/copper-gold-ratio-as-an-indicator.html",
                "Copper vs Gold. Copper represents industrial growth; Gold represents fear/inflation. Rising = Good Growth."
            )
            if sig: signals.append(sig)

        return signals


    def calculate_sector_turbulence(self, returns: pd.DataFrame, sector_name: str = "AI & Growth") -> float:
        """Calculate turbulence score specifically for a sector subset."""
        if sector_name not in self.ASSET_UNIVERSE:
            return 0.0
            
        sector_tickers = self.ASSET_UNIVERSE[sector_name]
        # Filter columns that exist in returns
        valid_tickers = [t for t in sector_tickers if t in returns.columns]
        
        if len(valid_tickers) < 5:
            return 0.0
            
        subset_returns = returns[valid_tickers]
        
        # Calculate for latest date
        try:
            # Use raw Mahalanobis (returns tuple, get first element)
            score, _ = self.calculate_turbulence(subset_returns, subset_returns.index[-1])
            return score
        except Exception as e:
            print(f"Sector turbulence error: {e}")
            return 0.0
