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
            "XLI", "XLB", "XLRE", "^VIX", "^VXV", "^GSPC", "ES=F"
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
            log_returns = np.log(prices / prices.shift(1)).dropna()
            
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
        dollar_vol = dollar_vol.replace(0, np.nan).fillna(method='ffill')
        
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

    def get_vix_term_structure_signal(self, prices: pd.DataFrame) -> str:
        """Check Backwardation (Spot > 3M)."""
        if '^VIX' in prices.columns and '^VXV' in prices.columns:
            spot = prices['^VIX'].iloc[-1]
            term = prices['^VXV'].iloc[-1]
            if term == 0: return "N/A"
            ratio = spot / term
            if ratio > 1.0:
                return "BACKWARDATION (Panic)"
            return "Contango (Normal)"
        return "N/A"

    def get_advanced_signal(self, turbulence: float, liquidity_z: float, hurst: float) -> str:
        """The Super-Signal logic."""
        if hurst > 0.75 and liquidity_z > 2.0:
            return "FRAGILE: Liquidity Hole Forming"
        
        if turbulence > 900:
            return "CRASH: Structural Break"
            
        if turbulence < 800 and liquidity_z < 1.0 and hurst < 0.5:
            return "BUY_SIGNAL: Liquidity Restored + Mean Reversion"
            
        return "NORMAL"

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

    def get_current_metrics(self, returns: pd.DataFrame, prices: pd.DataFrame = None, volumes: pd.DataFrame = None) -> MarketMetrics:
        """
        Calculate all current market metrics.
        
        Args:
            returns: DataFrame of log returns
            prices: DataFrame of prices (optional, for advanced metrics)
            volumes: DataFrame of volumes (optional, for advanced metrics)
            
        Returns:
            MarketMetrics containing all current values
        """
        # Get turbulence and contributions
        turbulence, contributions = self.calculate_turbulence(returns)
        
        # Get previous day turbulence for signal
        prev_turbulence = None
        if len(returns) > 1:
            try:
                prev_date = returns.index[-2]
                prev_turbulence, _ = self.calculate_turbulence(returns, prev_date)
            except Exception:
                pass # If calculation fails for prev_date, prev_turbulence remains None
        
        # Get absorption ratio
        absorption = self.calculate_absorption_ratio(returns)
        
        # Get SPY return
        spy_return = 0.0
        if 'SPY' in returns.columns:
            spy_return = returns['SPY'].iloc[-1] * 100  # Convert to percentage
        
        # Generate primary signal
        signal, message = self.generate_signal(
            spy_return, turbulence, absorption, len(returns.columns), prev_turbulence
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
            # Hurst (on SPY log prices)
            if "SPY" in prices.columns:
                 log_p = np.log(prices["SPY"].values)
                 hurst = self.get_hurst_exponent(log_p)
            
            # Liquidity
            liquidity_z = self.check_liquidity_stress(returns, prices, volumes, "SPY")
            
            # Advanced Signal
            adv_signal = self.get_advanced_signal(turbulence, liquidity_z, hurst)

        return MarketMetrics(
            turbulence_score=turbulence,
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
        Identify assets driving the current turbulence based on Z-Scores.
        """
        if len(returns) < 2:
            return []
            
        # Latest Returns
        latest = returns.iloc[-1]
        
        # Calculate recent volatility (Rolling 60d or full window if smaller)
        window = min(len(returns), 60)
        volatility = returns.rolling(window=window).std().iloc[-1]
        
        # Handle zero volatility to avoid division by zero
        volatility = volatility.replace(0, np.inf) 
        
        # Z-Scores (Standard Deviations moved)
        z_scores = latest / volatility
        
        # Create DataFrame for sorting
        scores = pd.DataFrame({
            'Ticker': z_scores.index,
            'Z_Score': z_scores.values,
            'Return': latest.values,
            'Abs_Z': np.abs(z_scores.values)
        }).sort_values('Abs_Z', ascending=False)
        
        # Format results
        drivers = []
        for _, row in scores.head(top_n).iterrows():
            drivers.append({
                'ticker': row['Ticker'],
                'z_score': row['Z_Score'],
                'return': row['Return'] * 100 # Convert to %
            })
            
        return drivers

    def get_macro_signals(self, returns: pd.DataFrame) -> List[Dict]:
        """
        Analyze macro-economic ratios for institutional recommendations.
        Uses cumulative returns to approximate price trends.
        """
        signals = []
        if len(returns) < 20:
            return signals
            
        # Reconstruct cumulative returns (Index starts at 1.0)
        cum_ret = np.exp(returns.cumsum())
        
        # Helper to check trend
        def check_trend(series, name, bullish_msg, bearish_msg):
            # Simple 20-day trend (Linear Regression slope or simple Start/End)
            recent = series.iloc[-20:]
            # Slope of normalized line
            y = recent.values
            x = np.arange(len(y))
            slope, _, _, _, _ = stats.linregress(x, y)
            
            trend_strength = slope * 100 # Scale for readability
            
            if trend_strength > 0.05:
                return {"pair": name, "trend": "Rising", "signal": bullish_msg, "strength": trend_strength}
            elif trend_strength < -0.05:
                return {"pair": name, "trend": "Falling", "signal": bearish_msg, "strength": trend_strength}
            return None

        # 1. EM vs US (EEM / SPY)
        if 'EEM' in cum_ret.columns and 'SPY' in cum_ret.columns:
            ratio = cum_ret['EEM'] / cum_ret['SPY']
            sig = check_trend(ratio, "EEM/SPY (EM vs US)", 
                            "Overweight Emerging Markets / Underweight US",
                            "US Exceptionalism Dominating (Stay Long US)")
            if sig: signals.append(sig)

        # 2. Risk Preference (SPY / TLT)
        if 'SPY' in cum_ret.columns and 'TLT' in cum_ret.columns:
            ratio = cum_ret['SPY'] / cum_ret['TLT']
            sig = check_trend(ratio, "SPY/TLT (Stocks vs Bonds)", 
                            "Risk-On: Equity Outperformance",
                            "Risk-Off: Flight to Quality (Bonds)")
            if sig: signals.append(sig)

        # 3. Small Cap Strength (IWM / SPY)
        if 'IWM' in cum_ret.columns and 'SPY' in cum_ret.columns:
            ratio = cum_ret['IWM'] / cum_ret['SPY']
            sig = check_trend(ratio, "IWM/SPY (Small Caps)", 
                            "Broadening Rally (Bullish Breadth)",
                            "Narrow Rally (Mega-Cap Dominance)")
            if sig: signals.append(sig)
            
        # 4. Safe Haven (GLD / SPY)
        if 'GLD' in cum_ret.columns and 'SPY' in cum_ret.columns:
            ratio = cum_ret['GLD'] / cum_ret['SPY']
            sig = check_trend(ratio, "GLD/SPY (Gold vs Stocks)", 
                            "Defensive Rotation: Gold Outperforming",
                            "Growth Focus: Gold Lagging")
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
