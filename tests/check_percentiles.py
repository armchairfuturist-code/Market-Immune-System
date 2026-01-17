import pandas as pd
import numpy as np
import config
from core import data_loader, math_engine

def check_percentiles():
    print("--- PERCENTILE CHECK ---")
    df_close, _ = data_loader.fetch_market_data(config.ASSET_UNIVERSE, period="3y")
    
    if df_close.empty:
        print("No data.")
        return
        
    turb = math_engine.calculate_turbulence(df_close)
    
    p95 = turb.quantile(0.95)
    p99 = turb.quantile(0.99)
    max_val = turb.max()
    
    print(f"P95: {p95:.2f}")
    print(f"P99: {p99:.2f}")
    print(f"Max: {max_val:.2f}")
    
    # Check what 280 corresponds to
    percentile_280 = (turb < 280).mean()
    print(f"280 is P{percentile_280*100:.1f}")

if __name__ == "__main__":
    check_percentiles()
