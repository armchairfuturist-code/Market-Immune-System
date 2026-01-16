import time
import pandas as pd
import numpy as np
from core import math_engine

def run_performance_test():
    print("Starting Performance Test...")
    
    # 1. Generate Large Dataset
    # 5 years of data for 100 assets
    days = 252 * 5
    assets = 100
    print(f"Generating data: {days} days x {assets} assets...")
    
    dates = pd.date_range(start="2020-01-01", periods=days)
    data = np.random.randn(days, assets)
    prices = pd.DataFrame(100 + np.cumsum(data, axis=0), index=dates, columns=[f"A{i}" for i in range(assets)])
    
    # 2. Test Turbulence Calculation
    start_time = time.time()
    math_engine.calculate_turbulence(prices, lookback=365)
    turb_time = time.time() - start_time
    print(f"Turbulence Calculation Time: {turb_time:.4f} seconds")
    
    # 3. Test Absorption Ratio
    start_time = time.time()
    math_engine.calculate_absorption_ratio(prices, window=60)
    abs_time = time.time() - start_time
    print(f"Absorption Ratio Time: {abs_time:.4f} seconds")
    
    # Thresholds (Example: expect < 5 seconds for acceptable UX)
    if turb_time > 5.0 or abs_time > 5.0:
        print("WARNING: Performance is slow!")
    else:
        print("Performance is acceptable.")

if __name__ == "__main__":
    run_performance_test()
