import pandas as pd
import numpy as np
import config

def detect_market_cycle(prices_df):
    """
    Determines the current market cycle phase based on Sector Relative Strength.
    
    Phases & Leaders:
    - Early Cycle: Financials (XLF), Real Estate (XLRE), Discretionary (XLY), Industrials (XLI)
    - Mid Cycle: Tech (XLK), Comm Services (XLC - approx by meta/goog or just XLK proxy)
    - Late Cycle: Energy (XLE), Materials (XLB)
    - Recession: Staples (XLP), Healthcare (XLV), Utilities (XLU)
    
    Logic:
    Calculate 3-month (60-day) Relative Strength of each sector vs SPY.
    The Phase is determined by which group has the highest aggregate RS.
    """
    if "SPY" not in prices_df.columns:
        return "Unknown", {}

    spy = prices_df["SPY"]
    
    # Define Sector Groups
    # Using available tickers in ASSET_UNIVERSE
    cycle_map = {
        "Phase I: Early Cycle (Recovery)": ["XLF", "XLRE", "XLY", "XLI"],
        "Phase II: Mid-Cycle (Expansion)": ["XLK", "XLC"], # XLC might need manual add if not in config
        "Phase III: Late Cycle (Slowdown)": ["XLE", "XLB"],
        "Phase IV: Recession (Contraction)": ["XLP", "XLV", "XLU"]
    }
    
    scores = {}
    
    for phase, tickers in cycle_map.items():
        # Filter for tickers present in data
        valid_tickers = [t for t in tickers if t in prices_df.columns]
        
        if not valid_tickers:
            scores[phase] = -999
            continue
            
        # Calculate Average RS Z-Score or Raw RS Performance?
        # Simple RS Performance: (Sector/SPY)_current / (Sector/SPY)_60d_ago - 1
        
        phase_scores = []
        for t in valid_tickers:
            sector_price = prices_df[t]
            rs = sector_price / spy
            
            # 60-day ROC of RS (approx 3 months)
            # Use rolling mean to smooth?
            # Let's take current RS vs 60d MA of RS to detect trend?
            # Or just ROC. ROC is standard.
            
            # Check length
            if len(rs) > 60:
                roc = (rs.iloc[-1] / rs.iloc[-60]) - 1
                phase_scores.append(roc)
        
        if phase_scores:
            scores[phase] = np.mean(phase_scores)
        else:
            scores[phase] = -999

    # Determine Winner
    # Sort scores
    sorted_phases = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    current_phase = sorted_phases[0][0]
    
    # Context data
    details = {
        "leading_phase": current_phase,
        "scores": scores,
        "top_sectors": sorted_phases[0][1] # Score of the winner
    }
    
    return current_phase, details
