import time
import pandas as pd
import numpy as np
from core import math_engine
import gc

def run_ceiling_test(total_points=10_000_000):
    print(f"--- Stress Test: Ceiling {total_points/1e6:.1f}M Points ---")
    
    # Estimate assets and days to hit the ceiling
    assets_count = 1000
    days = 10000
    
    print(f"Config: {assets_count} assets x {days} days")
    
    try:
        # Generate data
        print("Generating data...")
        data = np.random.randn(days, assets_count).astype(np.float32) 
        dates = pd.date_range(start="1980-01-01", periods=days)
        prices = pd.DataFrame(100 + np.cumsum(data, axis=0), index=dates)
        
        # Free up data to save RAM
        del data
        gc.collect()
        
        # 1. Test Turbulence
        print("Testing Turbulence Engine...")
        start_time = time.time()
        # Using a smaller lookback for the 10M test to focus on overhead/scaling
        math_engine.calculate_turbulence(prices, lookback=252)
        turb_time = time.time() - start_time
        print(f"Turbulence Time: {turb_time:.2f}s")
        
        # 2. Test Absorption Ratio (This is PCA heavy)
        print("Testing Absorption Ratio Engine...")
        start_time = time.time()
        # Window of 60 is standard
        math_engine.calculate_absorption_ratio(prices, window=60)
        abs_time = time.time() - start_time
        print(f"Absorption Ratio Time: {abs_time:.2f}s")
        
        print("--- Stress Test Complete ---")
        print(f"Total processing time for 10M points: {turb_time + abs_time:.2f}s")
        
    except MemoryError:
        print("FAILED: Memory Error encountered at 10M ceiling.")
    except Exception as e:
        print(f"FAILED: Unexpected error: {e}")

if __name__ == "__main__":
    run_ceiling_test()
