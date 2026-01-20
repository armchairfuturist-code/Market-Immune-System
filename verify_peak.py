import sys
sys.path.append('.')

import pandas as pd
from core import data_loader, math_engine
import config
import datetime

# Fetch data from 2023-01-01 to cover 2024
start_date = datetime.date(2023, 1, 1)

try:
    df_close, _ = data_loader.fetch_market_data(config.ASSET_UNIVERSE, start_date=start_date)
    print(f"Data fetched: {len(df_close)} rows from {df_close.index[0]} to {df_close.index[-1]}")

    # Slice to 2024 for efficiency
    df_2024 = df_close.loc['2024-01-01':'2024-12-31']
    print(f"2024 data: {len(df_2024)} rows")

    # Run turbulence on 2024 data
    turb_series = math_engine.calculate_turbulence(df_2024)
    print(f"Turbulence calculated, length: {len(turb_series)}")

    # Find max
    max_turb = turb_series.max()
    max_date = turb_series.idxmax()
    print(f"Max turbulence in 2024: {max_turb:.2f} on {max_date}")

    # Check August 5, 2024
    aug5 = pd.Timestamp('2024-08-05')
    if aug5 in turb_series.index:
        aug5_turb = turb_series.loc[aug5]
        print(f"Turbulence on 2024-08-05: {aug5_turb:.2f}")
    else:
        print("2024-08-05 not in data")

    # Check around that date
    around_aug5 = turb_series.loc['2024-08-01':'2024-08-10']
    print("Turbulence around Aug 5, 2024:")
    print(around_aug5)

except Exception as e:
    print(f"Error: {e}")