import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.linalg import solve
import config

def calculate_turbulence(prices_df, lookback=365):
    """
    Calculates the Statistical Turbulence (Mahalanobis Distance) of the asset universe.
    Optimized for performance and robustness.
    """
    # 1. Data Sanitization
    # Drop assets with > 30% missing data to prevent row-killing
    missing_frac = prices_df.isna().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index
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
            dist_values[i] = np.sqrt(max(0, d_sq))
            
        except Exception:
            continue

    raw_series = pd.Series(dist_values, index=dates)
    raw_series = raw_series.reindex(prices_df.index)
    
    p99 = raw_series.quantile(config.TURBULENCE_PERCENTILE)
    if np.isnan(p99) or p99 == 0:
        p99 = 1.0
        
    scaled_score = (raw_series / p99) * config.TURBULENCE_ANCHOR
    return scaled_score.clip(0, 1000)

def calculate_absorption_ratio(prices_df, window=60):
    """
    Calculates the Absorption Ratio (Systemic Fragility).
    AR = Variance(Top 20% Eigenvectors) / Total Variance
    """
    # 1. Data Sanitization
    missing_frac = prices_df.isna().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index
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
