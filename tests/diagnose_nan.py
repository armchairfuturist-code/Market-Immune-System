import pandas as pd
import numpy as np
import config
from core import data_loader, math_engine
import sys

def diagnose():
    print("--- DIAGNOSTIC START ---")
    
    # 1. Check Data Loading
    print("1. Fetching Data...")
    try:
        df_close, df_vol = data_loader.fetch_market_data(config.ASSET_UNIVERSE, period="2y")
    except Exception as e:
        print(f"FATAL: Data fetch failed: {e}")
        return

    print(f"Close Shape: {df_close.shape}")
    print(f"Volume Shape: {df_vol.shape}")
    
    if df_close.empty:
        print("FATAL: Close Prices DataFrame is empty.")
        return

    # Check for NaNs in Input
    nan_counts = df_close.isna().sum().sum()
    print(f"Total NaNs in Close Data: {nan_counts}")
    if nan_counts > 0:
        print("Sample NaNs per column:")
        print(df_close.isna().sum()[df_close.isna().sum() > 0].head())
        
    # Check SPY specifically
    if "SPY" in df_close.columns:
        spy_len = df_close["SPY"].count()
        print(f"SPY Valid Count: {spy_len} / {len(df_close)}")
    else:
        print("FATAL: SPY not in columns.")

    # 2. Check Turbulence
    print("\n2. Testing Turbulence...")
    lookback = 365
    print(f"Lookback: {lookback}")
    
    try:
        # Step-by-step turbulence check
        returns = np.log(df_close / df_close.shift(1)).dropna()
        print(f"Returns Shape: {returns.shape}")
        
        if len(returns) < lookback:
            print(f"WARNING: Returns length ({len(returns)}) < Lookback ({lookback}). Output will be all NaN.")
        
        turb = math_engine.calculate_turbulence(df_close, lookback=lookback)
        print(f"Turbulence Series Shape: {turb.shape}")
        print(f"Turbulence NaNs: {turb.isna().sum()}")
        print(f"Turbulence Last Value: {turb.iloc[-1]}")
        
    except Exception as e:
        print(f"Turbulence Error: {e}")

    # 3. Check Hurst
    print("\n3. Testing Hurst (SPY)...")
    try:
        if "SPY" in df_close.columns:
            hurst = math_engine.calculate_hurst(df_close["SPY"])
            print(f"Hurst Last Value: {hurst.iloc[-1]}")
            print(f"Hurst NaNs: {hurst.isna().sum()}")
            # Check inputs to log
            if (df_close["SPY"] <= 0).any():
                print("FATAL: Negative or Zero prices in SPY!")
        else:
            print("Skipping Hurst (No SPY)")
    except Exception as e:
        print(f"Hurst Error: {e}")

    # 4. Check Liquidity (Amihud)
    print("\n4. Testing Liquidity (Amihud SPY)...")
    try:
        if "SPY" in df_close.columns and "SPY" in df_vol.columns:
            # Check Volume
            vol_zeros = (df_vol["SPY"] == 0).sum()
            print(f"SPY Volume Zeros: {vol_zeros}")
            
            amihud = math_engine.calculate_amihud(df_close["SPY"], df_vol["SPY"])
            print(f"Amihud Last Value: {amihud.iloc[-1]}")
            print(f"Amihud NaNs: {amihud.isna().sum()}")
            
            # Debug Amihud intermediate
            returns = df_close["SPY"].pct_change().abs()
            dollar_vol = df_close["SPY"] * df_vol["SPY"]
            illiquidity = returns / dollar_vol
            print(f"Illiquidity (Raw) Infinite/NaNs: {np.isinf(illiquidity).sum()} / {illiquidity.isna().sum()}")
            
            ami_rolling = illiquidity.rolling(window=20).mean()
            print(f"Amihud (Rolling 20) NaNs: {ami_rolling.isna().sum()}")
            
            z_score = (ami_rolling - ami_rolling.rolling(window=252).mean()) / ami_rolling.rolling(window=252).std()
            print(f"Amihud Z-Score NaNs: {z_score.isna().sum()}")
            print(f"Required Data Length for First Valid Z: 20 + 252 = 272. Actual Data Length: {len(df_close)}")

    except Exception as e:
        print(f"Liquidity Error: {e}")

    print("--- DIAGNOSTIC END ---")

if __name__ == "__main__":
    diagnose()
