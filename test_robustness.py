import pandas as pd
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from core import math_engine

def test_robustness():
    print("Testing math_engine robustness...")
    
    # Create a dummy DataFrame with dates
    dates = pd.date_range('2024-01-01', periods=500)
    
    # 1. Test empty DataFrame (no columns)
    empty_df = pd.DataFrame(index=dates)
    print("Testing calculate_turbulence with 0 columns...")
    try:
        turb = math_engine.calculate_turbulence(empty_df)
        print(f"Success! Result type: {type(turb)}, Mean: {turb.mean()}")
    except Exception as e:
        print(f"Failed! calculate_turbulence crashed with: {e}")

    # 2. Test DataFrame with all NaNs
    nan_df = pd.DataFrame(np.nan, index=dates, columns=['AAPL', 'MSFT'])
    print("\nTesting calculate_turbulence with 100% NaNs...")
    try:
        turb = math_engine.calculate_turbulence(nan_df)
        print(f"Success! Result type: {type(turb)}, Mean: {turb.mean()}")
    except Exception as e:
        print(f"Failed! calculate_turbulence crashed with: {e}")

    # 3. Test absorption ratio with 1 column
    single_col_df = pd.DataFrame(np.random.randn(500, 1), index=dates, columns=['SPY'])
    print("\nTesting calculate_absorption_ratio with 1 column...")
    try:
        ar = math_engine.calculate_absorption_ratio(single_col_df)
        print(f"Success! Result type: {type(ar)}, Is null?: {ar.isnull().all()}")
    except Exception as e:
        print(f"Failed! calculate_absorption_ratio crashed with: {e}")

    # 4. Test sector turbulence with invalid/missing assets
    df_with_spy = pd.DataFrame(np.random.randn(500, 1), index=dates, columns=['SPY'])
    print("\nTesting calculate_sector_turbulence with invalid sector list...")
    try:
        sturb = math_engine.calculate_sector_turbulence(df_with_spy, ["NON_EXISTENT_1", "NON_EXISTENT_2"])
        print(f"Success! Result type: {type(sturb)}, Mean: {sturb.mean()}")
    except Exception as e:
        print(f"Failed! calculate_sector_turbulence crashed with: {e}")

if __name__ == "__main__":
    test_robustness()
