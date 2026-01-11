"""
Market Immune System - Core Calculation Engine
Detects market fragility through Statistical Turbulence and Absorption Ratio metrics.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
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
class MarketMetrics:
    """Container for market health metrics."""
    turbulence_score: float
    absorption_ratio: float
    signal: SignalStatus
    signal_message: str
    top_contributors: List[Tuple[str, float]]
    spy_return: float


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
            "XLI", "XLB", "XLRE"
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
    
    def __init__(self, lookback_days: int = 365, fetch_days: int = 750, min_lookback: int = 60):
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
    
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch price data from yfinance and calculate log returns.
        
        Returns:
            DataFrame of log returns indexed by date
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
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
            # Filter out days with >90% zero returns (holidays/weekends artifacts)
            non_zero_pct = (log_returns != 0).mean(axis=1)
            log_returns = log_returns[non_zero_pct > 0.1]
            
            # Dynamically adjust lookback if insufficient data
            available_days = len(log_returns)
            if available_days < self.lookback_days:
                # Use 60% of available data, but at least min_lookback
                adjusted_lookback = max(self.min_lookback, int(available_days * 0.6))
                self._effective_lookback = adjusted_lookback
            else:
                self._effective_lookback = self.lookback_days
            
            return log_returns
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch market data: {str(e)}")
    
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
        
        # Calculate mean return vector
        mu = lookback_returns.mean().values
        
        # Calculate covariance with Ledoit-Wolf shrinkage
        try:
            shrinkage = CovarianceShrinkage(lookback_returns)
            cov_matrix = shrinkage.ledoit_wolf()
            
            # Invert covariance matrix
            try:
                cov_inv = np.linalg.inv(cov_matrix)
            except np.linalg.LinAlgError:
                # Fallback to pseudo-inverse
                cov_inv = np.linalg.pinv(cov_matrix)
                
        except Exception:
            # Ultimate fallback: standard covariance with pseudo-inverse
            cov_matrix = lookback_returns.cov().values
            cov_inv = np.linalg.pinv(cov_matrix)
        
        # Calculate Mahalanobis distance squared
        diff = current_return - mu
        mahal_sq = np.dot(np.dot(diff, cov_inv), diff)
        
        # Calculate contribution per asset (partial Mahalanobis)
        contributions = pd.Series(index=returns.columns, dtype=float)
        for i, col in enumerate(returns.columns):
            partial_diff = np.zeros_like(diff)
            partial_diff[i] = diff[i]
            partial_mahal = np.dot(np.dot(partial_diff, cov_inv), partial_diff)
            contributions[col] = partial_mahal
        
        # Normalize contributions to sum to total
        if contributions.sum() > 0:
            contributions = contributions / contributions.sum() * mahal_sq
        
        # Return Raw Mahalanobis Distance instead of CDF
        return float(mahal_sq), contributions
    
    def calculate_rolling_turbulence(self, returns: pd.DataFrame, ema_span: int = 3) -> pd.Series:
        """
        Calculate rolling turbulence scores with EMA smoothing.
        
        Args:
            returns: DataFrame of log returns
            ema_span: Span for exponential moving average smoothing
            
        Returns:
            Series of smoothed turbulence scores
        """
        turbulence_scores = []
        valid_dates = returns.index[self._effective_lookback:]
        
        for date in valid_dates:
            try:
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
    
    def generate_signal(
        self,
        spx_return: float,
        turbulence_score: float,
        absorption_score: float,
        n_assets: int,
        prev_turbulence: Optional[float] = None
    ) -> Tuple[SignalStatus, str]:
        """
        Generate market signal based on current metrics.
        
        Args:
            spx_return: Today's SPX return
            turbulence_score: Current turbulence score (Raw Mahalanobis)
            absorption_score: Current absorption ratio (0-1000)
            n_assets: Number of assets (Degrees of Freedom)
            prev_turbulence: Previous day's turbulence score
            
        Returns:
            Tuple of (SignalStatus, message_string)
        """
        # Calculate Chi-squared thresholds
        # df = n_assets
        threshold_75 = stats.chi2.ppf(0.75, df=n_assets)
        threshold_95 = stats.chi2.ppf(0.95, df=n_assets)
        threshold_99 = stats.chi2.ppf(0.99, df=n_assets)
        
        # Check for opportunity (mean reversion)
        if prev_turbulence is not None:
            if prev_turbulence > threshold_99 and turbulence_score < threshold_95:
                return SignalStatus.BLUE, "OPPORTUNITY: Mean reversion detected. Consider buying."
        
        # Check for crash (Extreme Turbulence + Drop)
        if spx_return < -1.0 and turbulence_score > threshold_99:
            return SignalStatus.BLACK, "CRASH: Severe drawdown with extreme turbulence."
        
        # Check for divergence (Rising Market + High Turbulence)
        if spx_return > 0 and turbulence_score > threshold_95:
            return SignalStatus.RED, "DIVERGENCE: Market rising on broken structure."
        
        # Check for fragility (Absorption)
        if absorption_score > 800:
            return SignalStatus.ORANGE, "FRAGILE: High market unification. Risk of cascade."
        
        # Check for elevated turbulence
        if turbulence_score > threshold_75:
            return SignalStatus.ORANGE, f"ELEVATED: Turbulence above 75th percentile ({threshold_75:.1f})."
        
        # Normal conditions
        return SignalStatus.GREEN, "NORMAL: Market structure intact."
    
    def get_current_metrics(self, returns: pd.DataFrame) -> MarketMetrics:
        """
        Calculate all current market metrics.
        
        Args:
            returns: DataFrame of log returns
            
        Returns:
            MarketMetrics containing all current values
        """
        # Get turbulence and contributions
        turbulence, contributions = self.calculate_turbulence(returns)
        
        # Get previous day turbulence for signal
        try:
            prev_date = returns.index[-2]
            prev_turbulence, _ = self.calculate_turbulence(returns, prev_date)
        except Exception:
            prev_turbulence = None
        
        # Get absorption ratio
        absorption = self.calculate_absorption_ratio(returns)
        
        # Get SPY return
        if 'SPY' in returns.columns:
            spy_return = returns['SPY'].iloc[-1] * 100  # Convert to percentage
        else:
            spy_return = 0.0
        
        # Generate signal
        signal, message = self.generate_signal(
            spy_return, turbulence, absorption, len(returns.columns), prev_turbulence
        )
        
        # Get top 5 contributors
        top_contributors = contributions.nlargest(5)
        top_contributors_list = [
            (ticker, float(value)) for ticker, value in top_contributors.items()
        ]
        
        return MarketMetrics(
            turbulence_score=turbulence,
            absorption_ratio=absorption,
            signal=signal,
            signal_message=message,
            top_contributors=top_contributors_list,
            spy_return=spy_return
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
