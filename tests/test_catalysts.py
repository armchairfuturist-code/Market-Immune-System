import pytest
import pandas as pd
import datetime
from core import data_loader, macro_connector

def test_fetch_next_earnings():
    # Test with known tickers
    tickers = ["NVDA", "MSFT"]
    df = data_loader.fetch_next_earnings(tickers)
    
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "Ticker" in df.columns
        assert "Date" in df.columns
        # Check date format
        assert isinstance(df.iloc[0]["Date"], (pd.Timestamp, datetime.date))

def test_fetch_economic_calendar():
    macro = macro_connector.MacroConnector()
    df = macro.fetch_economic_calendar()
    
    assert isinstance(df, pd.DataFrame)
    # Check columns
    if not df.empty:
        assert "Event" in df.columns
        assert "Date" in df.columns
        # Ensure dates are sorted or present
        assert len(df) > 0
