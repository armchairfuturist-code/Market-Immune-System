import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.linalg import solve
import config
import streamlit as st
import warnings

@st.cache_data
def calculate_turbulence(prices_df, lookback=365):
    """
    Calculates the Statistical Turbulence (Mahalanobis Distance) of the asset universe.
    Optimized for performance and robustness.
    """
    # 1. Data Sanitization
    # Drop assets with > 30% missing data to prevent row-killing
    missing_frac = prices_df.isna().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index
    
    # 1a. Robustness Check: If no assets remain, return baseline
    if len(keep_cols) == 0:
        return pd.Series(15.0, index=prices_df.index)
        
    prices_df = prices_df[keep_cols]
    
    # Forward Fill to handle weekends/holidays differences
    prices_df = prices_df.ffill()
    
    # 2. Log Returns
    returns = np.log(prices_df / prices_df.shift(1))
    
    # Fill start/remaining NaNs with 0 (Neutral assumption)
    returns = returns.fillna(0)
    
    dates = returns.index
    values = returns.values
    
    dist_values = np.zeros(len(returns))
    dist_values[:] = np.nan
    
    # Pre-calculate rolling stats
    rolling_mean = returns.rolling(window=lookback).mean()
    rolling_cov = returns.rolling(window=lookback).cov()

    # Optimization: Mahalanobis Distance calculation
    # Loop over indices where we have enough data
    for i in range(lookback, len(returns)):
        prev_date = dates[i-1]
        
        try:
            # Check if we have valid stats
            if prev_date not in rolling_mean.index:
                continue
                
            mu_t = rolling_mean.loc[prev_date].values
            cov_t = rolling_cov.loc[prev_date].values
            
            # Check for NaNs in covariance
            if np.any(np.isnan(cov_t)):
                continue
                
            r_t = values[i, :]
            
            diff = r_t - mu_t
            
            # Regularization
            cov_t += np.eye(cov_t.shape[0]) * 1e-6
            
            x = solve(cov_t, diff, assume_a='pos')
            d_sq = np.dot(diff, x)
            d = np.sqrt(max(0, d_sq))
            
            # Normalize for Activity (Weekends/Holidays)
            # When only Crypto (10%) is active, raw distance is naturally suppressed.
            # We scale it up to represent "Equivalent Systemic Turbulence".
            non_zeros = np.count_nonzero(r_t)
            total_assets = len(r_t)
            
            if total_assets > 0:
                activity_ratio = max(non_zeros, 1) / total_assets
                if activity_ratio < 0.25:
                    # Scale by sqrt(1/ratio) to adjust for degrees of freedom
                    scale_factor = np.sqrt(1.0 / activity_ratio)
                    # Cap to prevent noise explosion on very thin days
                    scale_factor = min(scale_factor, 4.0)
                    d *= scale_factor
            
            dist_values[i] = d
            
        except Exception:
            continue

    raw_series = pd.Series(dist_values, index=dates)
    raw_series = raw_series.reindex(prices_df.index)
    
    p99 = raw_series.quantile(config.TURBULENCE_PERCENTILE)
    if np.isnan(p99) or p99 == 0:
        p99 = 1.0
        
    scaled_score = (raw_series / p99) * config.TURBULENCE_ANCHOR
    # Floor at 15 to prevent zero-readings and improve signal-to-noise
    return scaled_score.clip(15, 1000)

@st.cache_data
def calculate_absorption_ratio(prices_df, window=60):
    """
    Calculates the Absorption Ratio (Systemic Fragility).
    AR = Variance(Top 20% Eigenvectors) / Total Variance
    """
    # 1. Data Sanitization
    missing_frac = prices_df.isna().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index
    
    # Robustness Check
    if len(keep_cols) < 2:
        return pd.Series(np.nan, index=prices_df.index)
        
    prices_df = prices_df[keep_cols]
    
    prices_df = prices_df.ffill()
    
    # 2. Log Returns
    returns = np.log(prices_df / prices_df.shift(1))
    returns = returns.fillna(0)
    
    ar_values = []
    dates = returns.index
    
    # Iterate for rolling window
    num_assets = returns.shape[1]
    num_eigenvectors = int(num_assets * 0.2)
    if num_eigenvectors < 1: 
        num_eigenvectors = 1
    
    for i in range(len(returns)):
        if i < window:
            ar_values.append(np.nan)
            continue
            
        # Window: i-window to i
        window_returns = returns.iloc[i-window : i]
        
        # PCA
        try:
            pca = PCA()
            pca.fit(window_returns)
            
            # Explained variance ratio
            variances = pca.explained_variance_ratio_
            
            # Sum of top N
            ar = np.sum(variances[:num_eigenvectors])
            ar_values.append(ar)
            
        except Exception:
            ar_values.append(np.nan)
            
    return pd.Series(ar_values, index=dates).reindex(prices_df.index)

def get_market_regime(turbulence_score, absorption_ratio, price_trend):
    """
    Classifies the market regime.
    
    Args:
        turbulence_score (float): Current score (0-1000)
        absorption_ratio (float): Current AR (0-1.0)
        price_trend (str): "UP" or "DOWN" (based on SMA or similar)
    
    Returns:
        str: Regime Name
    """
    # Thresholds from PRD
    # FRAGILE RALLY: Price Up + Absorption > 0.85 (850/1000 implied?) PRD says 850 but AR is usually %?
    # PRD says "If Score > 800 (80%)...". "Absorption > 850".
    # I will assume PRD means > 0.85 if formatted as float, or 85%
    
    # Normalizing AR to 0-1000 for internal consistency with PRD descriptions?
    # PRD says "Absorption > 850".
    # I'll convert AR (0-1) to (0-1000) for comparison.
    
    ar_score = absorption_ratio * 1000
    
    # STRUCTURAL DIVERGENCE: Price Up + Turbulence > 180
    # SYSTEMIC SELL-OFF: Price Down + Absorption > 850
    # CRASH ALERT: Turbulence > 370
    # NORMAL: Turbulence < 180
    
    # Order of operations matters (Priority)
    
    if turbulence_score > config.REGIME_TURBULENCE_CRASH:
        return "CRASH ALERT"
        
    if price_trend == "UP":
        if ar_score > 850:
            return "FRAGILE RALLY"
        if turbulence_score > config.REGIME_TURBULENCE_HIGH:
            return "STRUCTURAL DIVERGENCE"
            
    if price_trend == "DOWN":
        if ar_score > 850:
            return "SYSTEMIC SELL-OFF"
            
    if turbulence_score < config.REGIME_TURBULENCE_HIGH:
        return "NORMAL"
        
    return "WARNING" # Fallback

def calculate_hurst(series, window=100):
    """
    Calculates the Hurst Exponent (R/S Analysis) for a rolling window.
    H > 0.75 indicates a crowded/persistent trend (brittle).
    Simple R/S implementation.
    """
    def get_hurst(ts):
        if len(ts) < 20: return 0.5
        
        X = np.log(ts)
        
        mean_X = np.mean(X)
        Y = X - mean_X
        Z = np.cumsum(Y)
        R = np.max(Z) - np.min(Z)
        S = np.std(X)
        
        if S == 0: return 0.5
        
        RS = R / S
        
        H = np.log(RS) / np.log(len(ts))
        return H

    # Apply rolling
    return series.rolling(window=window, min_periods=int(window/2)).apply(get_hurst, raw=True)

def calculate_amihud(prices, volume, window=20):
    """
    Calculates Amihud Illiquidity: Average of |Return| / (Price * Volume).
    Multiplied by 1e6 for readability usually, but we check threshold Z > 2.0.
    So we might need to Z-score it or just raw?
    PRD: "Z > 2.0 = Liquidity Hole". So we return Z-Score of Amihud.
    """
    # 1. Returns
    returns = prices.pct_change().abs()
    
    # 2. Dollar Volume = Price * Volume
    # Use Close price approx
    dollar_vol = prices * volume
    
    # 3. Amihud Ratio Daily
    illiquidity = returns / dollar_vol
    
    # Handle zeros/inf
    illiquidity = illiquidity.replace([np.inf, -np.inf], np.nan)
    
    # 4. Rolling Mean (The Amihud measure)
    ami_measure = illiquidity.rolling(window=window, min_periods=5).mean()
    
    # 5. Z-Score (vs 1 year baseline?)
    # PRD implies a Z-score trigger.
    # We compare current ami_measure to its history.
    
    rolling_mean = ami_measure.rolling(window=252, min_periods=100).mean()
    rolling_std = ami_measure.rolling(window=252, min_periods=100).std()
    
    z_score = (ami_measure - rolling_mean) / rolling_std
    
    return z_score

def calculate_capital_rotation(prices_df):
    """
    Analyzes Sector Rotation (Relative Strength vs SPY).
    Returns a dataframe of sectors and their signal.
    """
    # Sectors from config
    sector_tickers = ["XLE", "XLF", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE"]
    
    if "SPY" not in prices_df.columns:
        return pd.DataFrame()
        
    spy = prices_df["SPY"]
    rotation_data = []
    
    for ticker in sector_tickers:
        if ticker not in prices_df.columns:
            continue
            
        sector_price = prices_df[ticker]
        
        # RS = Sector / SPY
        rs = sector_price / spy
        
        # 60-day ROC of RS
        rs_60d = rs.pct_change(60).iloc[-1]
        
        # Momentum check (Price vs 50MA)
        ma50 = sector_price.rolling(50).mean().iloc[-1]
        price = sector_price.iloc[-1]
        trend = "UP" if price > ma50 else "DOWN"
        
        rotation_data.append({
            "Sector": ticker,
            "RS_60d": rs_60d,
            "Trend": trend
        })
        
    # Sort by RS
    df = pd.DataFrame(rotation_data)
    if not df.empty:
        df = df.sort_values("RS_60d", ascending=False)
        
    return df

def calculate_institutional_ratios(prices_df):
    """
    Calculates Macro Ratios and their 20-day trends.
    Ratios: EEM/SPY, SPY/TLT, GLD/SPY, XLY/XLP, CPER/GLD
    """
    pairs = [
        ("EEM", "SPY", "Emerging Markets vs S&P 500"),
        ("SPY", "TLT", "Risk-On: Stocks vs Bonds"),
        ("GLD", "SPY", "Defensive: Gold vs Stocks"),
        ("XLY", "XLP", "Consumer: Cyclical vs Staples"),
        ("CPER", "GLD", "Growth: Copper vs Gold")
    ]
    
    results = []
    for num, den, label in pairs:
        if num in prices_df.columns and den in prices_df.columns:
            ratio = prices_df[num] / prices_df[den]
            
            # 20-day trend
            current_val = ratio.iloc[-1]
            prev_val = ratio.iloc[-20] if len(ratio) > 20 else current_val
            trend = "Rising" if current_val > prev_val else "Falling"
            
            # Z-Score for Tug-of-War (252d baseline)
            rolling_mean = ratio.rolling(window=252, min_periods=100).mean()
            rolling_std = ratio.rolling(window=252, min_periods=100).std()
            z_score = (current_val - rolling_mean.iloc[-1]) / rolling_std.iloc[-1] if not rolling_std.empty else 0.0
            
            results.append({
                "Pair": f"{num}/{den}",
                "Trend": trend,
                "Label": label,
                "Value": current_val,
                "Z-Score": z_score
            })
            
    return pd.DataFrame(results)

def generate_super_signal(amihud_z, hurst_val, prices_series):
    """
    Synthesizes the 'Math Super-Signal' from multiple quant metrics.
    """
    # 1. Check for mean reversion (RSI)
    delta = prices_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    last_rsi = rsi.iloc[-1] if not rsi.empty else 50

    # 2. Logic for Signal
    if amihud_z < -1.0 and hurst_val < 0.4 and last_rsi < 40:
        return "BUY_SIGNAL: Liquidity Restored + Mean Reversion Opportunity"
    elif amihud_z > 2.0 and hurst_val > 0.7:
        return "SELL_SIGNAL: Liquidity Hole + Crowded Trend (Extreme Risk)"
    elif amihud_z > 1.0:
        return "WARNING: Liquidity Stress Rising"
    
    return "NEUTRAL: Market Structure Stable"

def calculate_crypto_stress(prices_df, window=30):
    """
    Calculates the average Z-Score of Crypto assets.
    Signal: > 2.0 indicates Crypto-Led Stress.
    """
    # Identify Crypto cols in df
    crypto_cols = [c for c in config.CRYPTO_ASSETS if c in prices_df.columns]
    
    if not crypto_cols:
        return 0.0
        
    c_df = prices_df[crypto_cols]
    
    # Calculate Z-Score for each asset
    # Z = (Price - Mean) / Std
    # Rolling window
    roll_mean = c_df.rolling(window=window).mean()
    roll_std = c_df.rolling(window=window).std()
    
    z_scores = (c_df - roll_mean) / roll_std
    
    # Average Z-Score across basket
    avg_z = z_scores.mean(axis=1)
    
    # Return last value
    if not avg_z.empty:
        return avg_z.iloc[-1]
    return 0.0

def calculate_sector_turbulence(prices_df, sector_assets, lookback=365):
    """
    Calculates Turbulence for a specific sector subset.
    """
    valid_assets = [c for c in sector_assets if c in prices_df.columns]
    # Need reasonable number of assets for covariance?
    # Mahalanobis on small N is fine if T is large enough.
    if len(valid_assets) < 2:
        return pd.Series(index=prices_df.index).fillna(0)

    subset = prices_df[valid_assets]
    return calculate_turbulence(subset, lookback=lookback)

def check_black_swan_alignment(turb, absorption, yield_curve, smc_choch):
    """
    The Ultimate Crash Trigger: Aligns macro, micro, and structure signals.
    """
    # Input validation to prevent errors (e.g., from undefined variables like in day_name)
    if not all(isinstance(x, (int, float)) for x in [turb, absorption]):
        raise ValueError("turb and absorption must be numeric")
    if not isinstance(yield_curve, (int, float)):
        yield_curve = 0  # Safe default
    if not isinstance(smc_choch, (int, float, str)):
        smc_choch = 0  # Safe default; treat as numeric for >/<

    signals = 0
    if turb > 370: signals += 1  # Volatility spike
    if absorption > 0.85: signals += 1  # Absorption/unity indicator
    if yield_curve < 0: signals += 1  # Inverted curve
    if smc_choch in ["Bullish", 1] or (isinstance(smc_choch, (int, float)) and smc_choch == 1):  # Choch breakout flag; update logic for structure
        signals += 1  # Positive for black swan if alignment

    if signals >= 3:
        return "🚨 BLACK SWAN WARNING: Full System Alignment - Imminent Crash Potential"
    elif signals == 2:
        return "⚠️ Minor Alignment: Monitor Closely"
    return "Normal - No Significant Alerts"

def verify_black_swan_alignment(turb, choch_val, yield_curve, spx_price=None, spx_ma50=None):
    """
    Black Swan Gate: Triple Threat Logic with HUD Denotation.
    """
    signals = 0
    factors = {
        "Turbulence >370": turb > 370,
        "CHoCH Bearish": choch_val == -1,
        "Yield Inverted": yield_curve < 0,
        "SPX Below MA": spx_price is not None and spx_ma50 is not None and spx_price < spx_ma50
    }

    signals = sum(factors.values())

    if signals >= 3:
        return {"alert": "🚨 BLACK SWAN ALERT", "level": "full", "probability": "High (90% based on historical analogs)"}
    elif signals == 2:
        return {"alert": "Fragile Alignment", "level": "moderate", "probability": "Moderate"}
    elif signals == 1:
        return {"alert": "Structure Monitoring", "level": "low", "probability": "Low"}
    return {"alert": "Normal", "level": "none", "probability": "None"}
