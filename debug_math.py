from market_immune_system import MarketImmuneSystem
import pandas as pd
import numpy as np
from scipy import stats

def investigate_math():
    with open("debug_math_output.txt", "w") as f:
        f.write("Initializing System...\n")
        mis = MarketImmuneSystem(min_lookback=30) # Force robust init
        
        f.write("Fetching Data...\n")
        returns = mis.fetch_data()
        f.write(f"Data Shape: {returns.shape}\n")
        f.write(f"Last Date: {returns.index[-1]}\n")
        
        # Inspect the last 5 rows of returns
        f.write("\nLast 5 rows of returns (mean of absolute values):\n")
        f.write(str(returns.tail(5).abs().mean(axis=1)) + "\n")
        
        # Check if last row is all zeros
        last_row = returns.iloc[-1]
        if (last_row == 0).all():
            f.write("CRITICAL: Last row of returns is ALL ZEROS!\n")
        elif (last_row.abs() < 1e-6).mean() > 0.9:
             f.write("CRITICAL: >90% of assets have near-zero returns!\n")
        else:
            f.write(f"Last row non-zero count: {(last_row != 0).sum()} / {len(last_row)}\n")
            
        # Calculate Turbulence for the last few days
        f.write("\n--- Turbulence Diagnosis ---\n")
        dates_to_check = returns.index[-5:]
        
        for date in dates_to_check:
            try:
                f.write(f"\nAnalyzing {date.date()}...\n")
                
                # Reconstruct the calculation step-by-step
                date_idx = returns.index.get_loc(date)
                lookback = mis._effective_lookback
                lookback_returns = returns.iloc[date_idx - lookback:date_idx]
                current_return = returns.loc[date].values
                
                mu = lookback_returns.mean().values
                cov = lookback_returns.cov().values 
                
                diff = current_return - mu
                
                # Check magnitude of diff
                f.write(f"  Mean Abs Diff: {np.mean(np.abs(diff)):.6f}\n")
                f.write(f"  Max Abs Diff: {np.max(np.abs(diff)):.6f}\n")
                
                # Check Covariance diagonal (Variance)
                variances = np.diag(cov)
                f.write(f"  Mean Variance: {np.mean(variances):.8f}\n")
                
                # Mahalanobis
                raw_score, _ = mis.calculate_turbulence(returns, date)
                f.write(f"  Calculated Score (CDF*1000): {raw_score}\n")
                
                # Manual check of D^2
                cov_inv = np.linalg.pinv(cov)
                mahal_sq = np.dot(np.dot(diff, cov_inv), diff)
                f.write(f"  Manual D^2: {mahal_sq:.4f}\n")
                
                df = len(returns.columns)
                expected_d2 = df
                f.write(f"  Expected D^2 (approx DF={df}): {expected_d2}\n")
                
                cdf = stats.chi2.cdf(mahal_sq, df=df)
                f.write(f"  CDF: {cdf:.6f}\n")
                
            except Exception as e:
                f.write(f"Error for {date}: {e}\n")

if __name__ == "__main__":
    investigate_math()
