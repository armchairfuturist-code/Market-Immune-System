import pandas as pd
import numpy as np
import config
from core import data_loader, math_engine

def optimize():
    print("--- THRESHOLD OPTIMIZATION ---")
    
    # 1. Fetch 3 Years of Data
    print("Fetching 3y Data...")
    df_close, _ = data_loader.fetch_market_data(config.ASSET_UNIVERSE, period="3y")
    
    if df_close.empty:
        print("Data fetch failed.")
        return

    # 2. Calculate Metrics
    print("Calculating Turbulence...")
    turbulence = math_engine.calculate_turbulence(df_close)
    
    if "SPY" not in df_close.columns:
        print("SPY not found.")
        return
        
    spy = df_close["SPY"]
    ma50 = spy.rolling(50).mean()
    
    # Align
    common = turbulence.index.intersection(spy.index)
    turbulence = turbulence.loc[common]
    spy = spy.loc[common]
    ma50 = ma50.loc[common]
    
    # 3. Test Thresholds
    # Target: 20-30 spikes in 3 years (approx 750 trading days)
    # A "Spike" is a contiguous block? Or individual days?
    # User said "20-30 spikes". Typically means distinct events.
    # So we count *blocks* of days where condition is True.
    
    thresholds = [180, 200, 220, 250, 280, 300, 350, 370]
    
    print("\n--- RESULTS (3 Years) ---")
    print(f"Total Trading Days: {len(common)}")
    
    best_thresh = 180
    best_diff = 999
    
    for t in thresholds:
        # Condition: Turb > T AND Price > MA50
        mask = (turbulence > t) & (spy > ma50)
        
        # Count distinct blocks (events)
        # Shift mask to find edges (False -> True)
        starts = (mask & ~mask.shift(1).fillna(False)).sum()
        
        days_active = mask.sum()
        
        print(f"Threshold {t}: {starts} Spikes ({days_active} days total)")
        
        # Target 25
        diff = abs(starts - 25)
        if diff < best_diff:
            best_diff = diff
            best_thresh = t
            
    print(f"\nRecommended Threshold for ~25 Spikes: {best_thresh}")

if __name__ == "__main__":
    optimize()
