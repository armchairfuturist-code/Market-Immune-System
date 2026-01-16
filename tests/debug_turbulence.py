import pandas as pd
import numpy as np
import config
from core import data_loader, math_engine
import sys

def analyze_turbulence():
    print("--- TURBULENCE ANALYSIS ---")
    
    # 1. Fetch Data
    print("Fetching Market Data...")
    try:
        # Use a longer period to ensure we capture volatility
        df_close, _ = data_loader.fetch_market_data(config.ASSET_UNIVERSE, period="2y")
    except Exception as e:
        print(f"Fetch failed: {e}")
        return

    if df_close.empty:
        print("Data empty.")
        return
        
    # 2. Re-run Math Engine Logic (Internals)
    print("Recalculating Turbulence Internals...")
    
    # Copy-paste logic from math_engine.calculate_turbulence to inspect intermediates
    prices_df = df_close.copy()
    lookback = 365
    
    # Data Sanitization
    missing_frac = prices_df.isna().mean()
    keep_cols = missing_frac[missing_frac < 0.3].index
    print(f"Original Assets: {len(prices_df.columns)}")
    print(f"Kept Assets: {len(keep_cols)}")
    prices_df = prices_df[keep_cols]
    
    prices_df = prices_df.ffill()
    returns = np.log(prices_df / prices_df.shift(1))
    returns = returns.fillna(0)
    
    # Check Zeros
    total_cells = returns.size
    zero_cells = (returns == 0).sum().sum()
    print(f"Total Returns Cells: {total_cells}")
    print(f"Zero Returns Cells: {zero_cells} ({zero_cells/total_cells:.1%})")
    
    # Calculate Turbulence
    scores = math_engine.calculate_turbulence(df_close, lookback=lookback)
    
    # Analyze Scores
    print("\n--- SCORE STATISTICS ---")
    valid_scores = scores.dropna()
    print(f"Valid Scores Count: {len(valid_scores)}")
    
    if len(valid_scores) == 0:
        print("No valid scores.")
        return
        
    print(f"Min: {valid_scores.min():.2f}")
    print(f"Max: {valid_scores.max():.2f}")
    print(f"Mean: {valid_scores.mean():.2f}")
    print(f"Median: {valid_scores.median():.2f}")
    print(f"Last Value: {valid_scores.iloc[-1]:.2f}")
    
    # Reverse Engineer Raw Values
    # Scaled = (Raw / P99) * 370
    # Raw = (Scaled / 370) * P99
    # But we can just see what P99 was implied
    # Since we can't easily get 'Raw' from the function return (it returns scaled),
    # We infer P99 location.
    
    # Let's find the 99th percentile of the SCALED scores
    p99_scaled = valid_scores.quantile(config.TURBULENCE_PERCENTILE)
    print(f"P99 of Scaled Series (Should be ~370): {p99_scaled:.2f}")
    
    # Identify Outliers
    print("\n--- TOP 5 DAYS ---")
    print(valid_scores.sort_values(ascending=False).head(5))
    
    # Check Weekends
    # Add day name
    scored_df = valid_scores.to_frame(name="Score")
    scored_df["Day"] = scored_df.index.day_name()
    
    print("\n--- AVG SCORE BY DAY ---")
    print(scored_df.groupby("Day")["Score"].mean().sort_values())

if __name__ == "__main__":
    analyze_turbulence()
