import pytest
import pandas as pd
import numpy as np
from core import math_engine
import config

def test_get_market_regime():
    # Test CRASH ALERT
    assert math_engine.get_market_regime(400, 0.5, "UP") == "CRASH ALERT"
    
    # Test FRAGILE RALLY
    # Price Up + Absorption > 850 (normalized to 0-1000 in input? No, func takes 0-1 and multiplies)
    # Func logic: `ar_score = absorption_ratio * 1000`. If ar_score > 850...
    assert math_engine.get_market_regime(100, 0.9, "UP") == "FRAGILE RALLY"
    
    # Test STRUCTURAL DIVERGENCE
    # Price Up + Turbulence > 180
    assert math_engine.get_market_regime(200, 0.5, "UP") == "STRUCTURAL DIVERGENCE"
    
    # Test SYSTEMIC SELL-OFF
    # Price Down + Absorption > 850
    assert math_engine.get_market_regime(100, 0.9, "DOWN") == "SYSTEMIC SELL-OFF"
    
    # Test NORMAL
    assert math_engine.get_market_regime(50, 0.5, "UP") == "NORMAL"

def test_calculate_turbulence():
    # Create synthetic data: 100 days, 5 assets
    dates = pd.date_range(start="2023-01-01", periods=100)
    data = np.random.randn(100, 5)
    # Make prices positive
    prices = pd.DataFrame(100 + np.cumsum(data, axis=0), index=dates, columns=["A", "B", "C", "D", "E"])
    
    # Test
    # Lookback smaller than data
    turb = math_engine.calculate_turbulence(prices, lookback=20)
    
    assert len(turb) == 100
    assert isinstance(turb, pd.Series)
    assert turb.max() <= 1000
    assert turb.min() >= 0
    
    # Check that first 'lookback' are NaN or 0 (implementation details check)
    # My impl: initialize with NaN, loop starts at lookback.
    assert np.isnan(turb.iloc[0])
    assert not np.isnan(turb.iloc[-1])

def test_calculate_absorption_ratio():
    # Create synthetic highly correlated data
    dates = pd.date_range(start="2023-01-01", periods=100)
    # Asset A
    a = np.random.randn(100)
    # Asset B is almost exactly A (high correlation)
    b = a * 1.1 + np.random.normal(0, 0.01, 100)
    
    prices = pd.DataFrame({
        "A": 100 + np.cumsum(a),
        "B": 100 + np.cumsum(b)
    }, index=dates)
    
    ar = math_engine.calculate_absorption_ratio(prices, window=20)
    
    assert len(ar) == 100
    # High correlation should yield high AR (approaching 1.0)
    # AR for 2 assets, 1st eigenvector explains most variance if correlated.
    # Top 20% of 2 assets is 0.4 assets -> min 1 eigenvector.
    # If perfectly correlated, AR ~ 1.0.
    
    last_ar = ar.iloc[-1]
    assert not np.isnan(last_ar)
    assert 0.0 <= last_ar <= 1.0
    # With high correlation, AR should be high (> 0.8)
    # Check if it is relatively high
    assert last_ar > 0.5 
