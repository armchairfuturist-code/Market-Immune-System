import sys
sys.path.append('.')

import streamlit as st
from core import data_loader, math_engine, macro_connector, cycle_engine, report_generator, cycle_playbook, smc_engine
from ui import charts
import pandas as pd
import datetime
import config

# Simulate the data loading
start_date = datetime.datetime.now().date() - datetime.timedelta(days=730)
analysis_date = datetime.datetime.now().date()

buffer_date = start_date - datetime.timedelta(days=365)

try:
    df_close, hourly_df = data_loader.fetch_market_data(config.ASSET_UNIVERSE[:5], start_date=buffer_date)  # Limit to 5 for test
    print(f"Data fetched: {len(df_close)} rows, {len(df_close.columns)} columns")
    print(f"Columns: {df_close.columns.tolist()[:10]}")

    # Run math
    turb_series = math_engine.calculate_turbulence(df_close)
    print(f"Turbulence last: {turb_series.iloc[-1] if not turb_series.empty else 'Empty'}")

    if "SPY" in df_close.columns:
        hurst_series = math_engine.calculate_hurst(df_close["SPY"])
        print(f"Hurst last: {hurst_series.iloc[-1] if not hurst_series.empty else 'Empty'}")

    # Slice
    curr_turb = turb_series.loc[start_date:analysis_date]
    print(f"Curr turb last: {curr_turb.iloc[-1] if not curr_turb.empty else 'Empty'}")

except Exception as e:
    print(f"Error: {e}")