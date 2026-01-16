"""
Market Immune System - Core Calculation Engine v2.0
Detects market fragility, structural breaks, and tactical regimes.
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
from datetime import datetime, timedelta

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
    v2.0: Robust Data Ingestion, Calibrated Metrics, Regime Detection.
    """
    
    # Asset Universe - The "99" Index
    ASSET_UNIVERSE = {
        "Broad Markets": [
            "SPY", "QQQ", "DIA", "IWM", "VXX", "EEM", "EFA", "TLT", "IEF", "SHY",
            "LQD", "HYG", "BND", "AGG", "GLD", "SLV", "CPER", "USO", "UNG", "DBC",
            "PALL", "UUP", "FXE", "FXY", "FXB", "CYB", "XLF", "XLE", "XLK", "XLY",
            "XLI", "XLB", "XLRE", "^VIX", "^VIX3M"
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
        self.lookback_days = lookback_days
        self.fetch_days = fetch_days
        self.min_lookback = min_lookback
        self._effective_lookback = lookback_days
        self._all_tickers = self._get_all_tickers()
        
    def _get_all_tickers(self) -> List[str]:
        tickers = []
        for group in self.ASSET_UNIVERSE.values():
            tickers.extend([t.strip() for t in group])
        return list(set(tickers)) # Deduplicate
    
    def fetch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Robust Data Fetching Pipeline (v2.0).
        Implements 'Monday Fix' (Outer Join) and 'Today Fix' (Live Candle).
        """
        try:
            # 1. Fetch Historical Data
            data = yf.download(
                self._all_tickers,
                period=f"{self.fetch_days}d",
                auto_adjust=True,
                progress=False,
                threads=True
            )
            
            # 2. Extract Price & Volume
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data
                volume = data['Volume'] if 'Volume' in data.columns.get_level_values(0) else pd.DataFrame(index=data.index)
            else:
                prices = data['Close']
                volume = data['Volume']

            # 3. LIVE DATA APPEND (The "Today" Fix)
            try:
                today = datetime.now().date()
                last_date = prices.index[-1].date()
                
                # Check if market might be open/active today
                if last_date < today:
                    live = yf.download(self._all_tickers, period="1d", progress=False, auto_adjust=True)
                    if not live.empty:
                         # Handle MultiIndex for live data
                        if isinstance(live.columns, pd.MultiIndex):
                            live_p = live['Close'] if 'Close' in live.columns else live
                            live_v = live['Volume'] if 'Volume' in live.columns else pd.DataFrame()
                        else:
                            live_p = live
                            live_v = live['Volume'] if 'Volume' in live.columns else pd.DataFrame()
                        
                        # Align and Concat
                        live_p = live_p[prices.columns.intersection(live_p.columns)]
                        # Use concat with outer join logic implicitly via axis=0
                        if not live_p.empty and live_p.index[-1] not in prices.index:
                            prices = pd.concat([prices, live_p]).sort_index()
                            # Only concat volume if available
                            if not live_v.empty:
                                live_v = live_v[volume.columns.intersection(live_v.columns)]
                                volume = pd.concat([volume, live_v]).sort_index()
            except Exception as e:
                print(f"Live data append warning: {e}")

            # 4. THE MONDAY FIX: Handle Crypto/Stock Alignment
            # Forward fill to propagate Friday closes through the weekend for crypto calculation
            # This ensures Crypto (24/7) doesn't get dropped when merging with Stocks (M-F)
            prices = prices.ffill()
            volume = volume.ffill()

            # 5. Clean & Return
            # Calculate log returns
            log_returns = np.log(prices / prices.shift(1))
            log_returns = log_returns.replace([np.inf, -np.inf], np.nan).dropna(how='all')
            
            # Ensure valid index alignment
            common_idx = log_returns.index
            prices = prices.reindex(common_idx)
            volume = volume.reindex(common_idx)
            
            # Adjust lookback if data is short
            if len(log_returns) < self.lookback_days:
                self._effective_lookback = max(self.min_lookback, int(len(log_returns) * 0.6))
                
            return log_returns, prices, volume

        except Exception as e:
            print(f"CRITICAL DATA FETCH ERROR: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def calculate_turbulence(self, returns: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> Tuple[float, pd.Series]:
        """
        Calculate Statistical Turbulence (Mahalanobis Distance).
        Uses Winsorization + Ledoit-Wolf Shrinkage + Pseudo-Inverse.
        """
        if target_date is None: target_date = returns.index[-1]
        
        # Determine Window
        date_idx = returns.index.get_loc(target_date)
        lookback = self._effective_lookback
        if date_idx < lookback: lookback = max(self.min_lookback, date_idx)
        
        lookback_returns = returns.iloc[date_idx - lookback:date_idx]
        current_return = returns.loc[target_date].values
        
        try:
            # 1. Winsorization (Robustness)
            clipped = mstats.winsorize(lookback_returns.values, limits=[0.025, 0.025], axis=0)
            df_clipped = pd.DataFrame(clipped, columns=lookback_returns.columns)
            mu = df_clipped.mean().values
            
            # 2. Covariance Shrinkage (Ledoit-Wolf)
            shrink = CovarianceShrinkage(df_clipped)
            cov = shrink.ledoit_wolf()
            
            # 3. Regularization & Inversion
            epsilon = 1e-4
            cov += np.eye(len(lookback_returns.columns)) * epsilon
            inv_cov = np.linalg.pinv(cov)
            
            # 4. Mahalanobis Distance
            diff = current_return - mu
            mahal_sq = np.dot(np.dot(diff, inv_cov), diff)
            
            # 5. Attribution
            contribs = pd.Series(index=returns.columns, dtype=float)
            for i, col in enumerate(returns.columns):
                d = np.zeros_like(diff); d[i] = diff[i]
                contribs[col] = np.dot(np.dot(d, inv_cov), d)
            
            if contribs.sum() > 0:
                contribs = contribs / contribs.sum() * mahal_sq
                
            return float(mahal_sq), contribs
            
        except Exception:
            return 0.0, pd.Series(dtype=float)

    def calculate_rolling_turbulence(self, returns: pd.DataFrame, ema_span: int = 3) -> pd.Series:
        """Rolling Turbulence with Expanding Window."""
        scores = []
        # Start from min_lookback
        valid_dates = returns.index[self.min_lookback:]
        
        for date in valid_dates:
            try:
                score, _ = self.calculate_turbulence(returns, date)
                scores.append(score)
            except:
                scores.append(np.nan)
                
        return pd.Series(scores, index=valid_dates).ewm(span=ema_span, adjust=False).mean()

    def calculate_absorption_ratio(self, returns: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> float:
        """Calculate Absorption Ratio (PCA Variance)."""
        if target_date is None: target_date = returns.index[-1]
        date_idx = returns.index.get_loc(target_date)
        lookback = self._effective_lookback
        if date_idx < lookback: lookback = max(self.min_lookback, date_idx)
        
        subset = returns.iloc[date_idx - lookback:date_idx]
        if len(subset) < 10: return 0.0
        
        pca = PCA()
        pca.fit(subset.corr().fillna(0))
        var = pca.explained_variance_ratio_
        top_n = max(1, int(len(var) * 0.20))
        
        return (var[:top_n].sum() / var.sum()) * 1000

    def calculate_rolling_liquidity(self, returns: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, ticker: str = "SPY") -> pd.Series:
        """Rolling Amihud Illiquidity Z-Score (Vectorized)."""
        if ticker not in returns.columns: return pd.Series(0.0, index=returns.index)
        
        r = returns[ticker].abs()
        p = prices[ticker]
        v = volumes[ticker]
        dollar_vol = (p * v).replace(0, np.nan).ffill()
        
        illiq = r / dollar_vol
        illiq_smooth = illiq.rolling(10).mean()
        
        mu = illiq_smooth.rolling(365).mean()
        sigma = illiq_smooth.rolling(365).std()
        
        return ((illiq_smooth - mu) / sigma).fillna(0.0)

    def check_liquidity_stress(self, returns: pd.DataFrame, prices: pd.DataFrame, volumes: pd.DataFrame, ticker: str = "SPY") -> float:
        """Latest Liquidity Z-Score."""
        s = self.calculate_rolling_liquidity(returns, prices, volumes, ticker)
        return s.iloc[-1] if not s.empty else 0.0

    @staticmethod
    def get_hurst_exponent(time_series: np.array, max_lag: int = 20) -> float:
        """Hurst Exponent Calculation."""
        try:
            lags = range(2, max_lag)
            tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
            m = np.polyfit(np.log(lags), np.log(tau), 1)
            return m[0]
        except: return 0.5

    def get_futures_sentiment(self) -> Dict:
        """Futures vs Spot Basis."""
        try:
            tickers = ["^GSPC", "ES=F", "BTC-USD", "BTC=F"]
            data = yf.download(tickers, period="5d", progress=False)["Close"].iloc[-1]
            
            spx_s, spx_f = data.get("^GSPC", 0), data.get("ES=F", 0)
            btc_s, btc_f = data.get("BTC-USD", 0), data.get("BTC=F", 0)
            
            spx_b = ((spx_f - spx_s) / spx_s * 100) if spx_s > 0 else 0
            btc_b = ((btc_f - btc_s) / btc_s * 100) if btc_s > 0 else 0
            
            return {
                "spx_basis": spx_b,
                "spx_signal": "BEARISH (Backwardation)" if spx_b < -0.02 else "NORMAL",
                "btc_basis": btc_b,
                "btc_signal": "INSTITUTIONAL SHORT" if btc_b < -0.5 else "NORMAL"
            }
        except: return {}

    def get_vix_term_structure_signal(self, prices: pd.DataFrame) -> str:
        """VIX Term Structure (VIX3M / VIX)."""
        try:
            spot = prices['^VIX'].iloc[-1]
            term = prices['^VIX3M'].iloc[-1] if '^VIX3M' in prices else prices.get('^VXV', pd.Series([1])).iloc[-1]
            ratio = spot / term if term > 0 else 1.0
            return f"BACKWARDATION {ratio:.2f}" if ratio > 1.0 else f"Contango {ratio:.2f}"
        except: return "N/A"

    def get_market_cycle_status(self, returns: pd.DataFrame) -> Dict:
        """Determine Economic Cycle Phase."""
        cycles = {
            "Early Cycle": {"t": ["XLF", "XLY", "IWM"], "d": "Financials, Discretionary", "n": "Recovery. Risk-On."},
            "Mid Cycle": {"t": ["XLK", "XLI"], "d": "Tech, Industrials", "n": "Growth Peak."},
            "Late Cycle": {"t": ["XLE", "XLB", "XLP"], "d": "Energy, Staples", "n": "Inflationary/Defensive."},
            "Recession": {"t": ["XLU", "XLV", "GLD"], "d": "Utilities, Gold", "n": "Contraction. Safety."}
        }
        if len(returns) < 60: return {}
        
        # Relative Strength vs SPY (60d)
        spy = np.exp(returns["SPY"].tail(60).cumsum().iloc[-1]) - 1 if "SPY" in returns else 0
        scores = {}
        
        for phase, info in cycles.items():
            valid = [t for t in info['t'] if t in returns]
            if valid:
                perfs = [(np.exp(returns[t].tail(60).cumsum().iloc[-1]) - 1) for t in valid]
                scores[phase] = (np.mean(perfs) - spy) * 100
        
        if not scores: return {}
        best = max(scores, key=scores.get)
        return {
            "current_phase": best,
            "strength": scores[best],
            "actionable_tickers": cycles[best]['d'],
            "narrative": cycles[best]['n']
        }
    
    def get_narrative_battle(self, returns: pd.DataFrame) -> Dict:
        """Compare AI vs Crypto performance (5d)."""
        if len(returns) < 5: return {}
        
        ai_t = ["NVDA", "AMD", "PLTR", "SMCI"]
        cry_t = ["BTC-USD", "ETH-USD", "COIN", "MSTR"]
        
        def get_perf(tickers):
            valid = [t for t in tickers if t in returns]
            if not valid: return 0.0
            return np.mean([(np.exp(returns[t].tail(5).cumsum().iloc[-1]) - 1) * 100 for t in valid])
            
        ai_p = get_perf(ai_t)
        cry_p = get_perf(cry_t)
        
        return {
            "ai_perf": ai_p,
            "crypto_perf": cry_p,
            "leader": "AI Dominance" if ai_p > cry_p else "Crypto Speculation"
        }
        
    def calculate_crypto_zscore(self, returns: pd.DataFrame) -> Tuple[float, float]:
        """Check if Crypto is leading volatility."""
        if "BTC-USD" not in returns: return 0.0, 0.0
        btc = returns["BTC-USD"]
        z = (btc.iloc[-1] - btc.mean()) / btc.std()
        return z, btc.iloc[-1]

    def get_current_metrics(self, returns: pd.DataFrame, prices: pd.DataFrame = None, volumes: pd.DataFrame = None, target_date: pd.Timestamp = None) -> MarketMetrics:
        if target_date is None: target_date = returns.index[-1]
        
        # 1. Metrics
        turb_raw, contribs = self.calculate_turbulence(returns, target_date)
        abs_ratio = self.calculate_absorption_ratio(returns, target_date)
        spy_ret = returns.loc[target_date, 'SPY'] * 100 if 'SPY' in returns.columns else 0.0
        
        # 2. Calibration (P99 Anchor = 370)
        df_assets = len(returns.columns)
        p99_est = stats.chi2.ppf(0.99, df=df_assets)
        turb_calib = (turb_raw / p99_est) * 370
        
        # 3. Signals
        # Basic Signal
        sig, msg = self.generate_signal(spy_ret, turb_calib, abs_ratio, df_assets)
        
        # Advanced Signal Inputs
        hurst = 0.5
        liq_z = 0.0
        
        if prices is not None and "SPY" in prices:
            try:
                hist = prices.loc[:target_date, "SPY"].iloc[-300:]
                hurst = self.get_hurst_exponent(np.log(hist.values))
            except: pass
            
        if volumes is not None:
            liq_z = self.check_liquidity_stress(returns.loc[:target_date], prices.loc[:target_date], volumes.loc[:target_date], "SPY")

        # Advanced Signal Logic
        adv_sig = "Normal Market Conditions"
        if abs_ratio > 850:
            if turb_calib < 300 and liq_z < 1.0: adv_sig = "Caution: Fragile Buy (High Absorption)"
            else: adv_sig = "Warning: Market Locked in Lockstep"
        elif hurst > 0.75 and liq_z > 2.0:
            adv_sig = "Danger: Liquidity Hole Forming"
        elif turb_calib > 370:
            adv_sig = "Critical: Structural Break Detected"
        elif turb_calib < 180 and liq_z < 1.0 and hurst < 0.5:
            adv_sig = "Strong Buy: Liquidity Restored"

        # 4. Contributors (Type Safe)
        try:
            contribs = pd.to_numeric(contribs, errors='coerce').fillna(0.0)
            top_5 = contribs.nlargest(5)
            top_contributors = [(str(t), float(v)) for t, v in top_5.items()]
        except: top_contributors = []
        
        # 5. Days Elevated Counter
        days_elevated = 0
        # (Calculated in main via rolling series for efficiency, returning 0 here or pass series if needed)
        # We'll leave it as 0 here and let main handle the rolling count context.

        return MarketMetrics(
            turbulence_score=turb_calib,
            absorption_ratio=abs_ratio,
            signal=sig,
            signal_message=msg,
            top_contributors=top_contributors,
            spy_return=spy_ret,
            context=None,
            hurst_exponent=hurst,
            liquidity_z=liq_z,
            advanced_signal=adv_sig
        )

    def generate_signal(self, spx_return: float, turbulence: float, absorption: float, n_assets: int, prev_turb: Optional[float] = None) -> Tuple[SignalStatus, str]:
        """Signal Logic based on Calibrated Thresholds (180/370)."""
        WARN = 180.0
        CRIT = 370.0
        
        if prev_turb and prev_turb > CRIT and turbulence < 300:
            return SignalStatus.BLUE, "OPPORTUNITY: Mean Reversion."
        if spx_return < -1.0 and turbulence > CRIT:
            return SignalStatus.BLACK, "CRASH: Liquidity Evaporation."
        if spx_return > 0 and turbulence > WARN:
            return SignalStatus.RED, "DIVERGENCE: Rising on Broken Structure."
        if absorption > 850:
            return SignalStatus.ORANGE, "FRAGILE: High Unification."
        if turbulence > WARN:
            return SignalStatus.ORANGE, f"ELEVATED: Turbulence > {WARN}."
            
        return SignalStatus.GREEN, "NORMAL: Structure Intact."

    def calculate_days_elevated(self, turbulence_series: pd.Series, threshold: float) -> int:
        if len(turbulence_series) == 0 or turbulence_series.iloc[-1] <= threshold: return 0
        count = 0
        for score in reversed(turbulence_series.values):
            if score > threshold: count += 1
            else: break
        return count
    
    def get_macro_signals(self, returns: pd.DataFrame, target_date: Optional[pd.Timestamp] = None) -> List[Dict]:
        """Macro Trends (GLD/SPY Reflation etc)."""
        if target_date is None: target_date = returns.index[-1]
        valid = returns.loc[:target_date]
        if len(valid) < 20: return []
        
        cum_ret = np.exp(valid.cumsum())
        signals = []
        
        def check(pair_name, num, den, bull, bear, url, desc):
            if num not in cum_ret or den not in cum_ret: return
            ratio = cum_ret[num] / cum_ret[den]
            recent = ratio.tail(20)
            slope = stats.linregress(np.arange(len(recent)), recent.values).slope * 100
            
            sig = { "pair": pair_name, "url": url, "desc": desc, "strength": slope }
            if slope > 0.05: sig.update({"trend": "Rising", "signal": bull})
            elif slope < -0.05: sig.update({"trend": "Falling", "signal": bear})
            else: return # Neutral
            signals.append(sig)

        # Context: GLD Rising?
        gld_rising = False
        if "GLD" in cum_ret and "SPY" in cum_ret:
             gld_r = cum_ret["GLD"] / cum_ret["SPY"]
             if stats.linregress(np.arange(20), gld_r.tail(20).values).slope > 0: gld_rising = True

        # 1. Risk Preference
        check("SPY/TLT", "SPY", "TLT", 
              "Reflationary Melt-Up" if gld_rising else "Risk-On: Stocks > Bonds",
              "Risk-Off: Flight to Bonds",
              "https://stockcharts.com/school/doku.php?id=chart_school:market_analysis:intermarket_analysis",
              "Stocks vs Bonds. Rising = Growth/Inflation.")
              
        # 2. Economic Cycle
        check("XLY/XLP", "XLY", "XLP",
              "Confident Consumer", "Defensive Posturing",
              "https://school.stockcharts.com/doku.php?id=market_analysis:sector_rotation_analysis",
              "Discretionary vs Staples. Rising = Economic Confidence.")
              
        # 3. Credit Stress
        check("HYG/LQD", "HYG", "LQD",
              "Credit Appetite", "Credit Stress (Spreads Widening)",
              "https://www.investopedia.com/terms/h/high_yield_bond.asp",
              "Junk vs Grade. Falling = Credit Freeze Risk.")
              
        # 4. Copper/Gold
        check("CPER/GLD", "CPER", "GLD",
              "Global Growth (Dr. Copper)", "Stagflation Risk (Gold)",
              "https://www.cmegroup.com/education/featured-reports/copper-gold-ratio-as-an-indicator.html",
              "Industrial vs Safe Haven. Rising = Real Growth.")
              
        return signals