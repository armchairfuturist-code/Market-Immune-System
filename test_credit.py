import sys
sys.path.append('.')

from core.macro_connector import MacroConnector
import pandas as pd

macro = MacroConnector()
credit = macro.fetch_credit_spreads()
print(f"Type: {type(credit)}")
print(f"Length: {len(credit)}")
if isinstance(credit, pd.Series) and not credit.empty:
    print(f"Last value: {credit.iloc[-1]}")
    print(f"Last 5 values: {credit.tail()}")
    # Check if nan
    if pd.isna(credit.iloc[-1]):
        print("Last value is NaN")
        # Check roll_mean and roll_std
        spreads = macro.fred.get_series('BAMLH0A0HYM2')
        print(f"Spreads date range: {spreads.index[0]} to {spreads.index[-1]}")
        print(f"Spreads na count: {spreads.isna().sum()}")
        print(f"Spreads length: {len(spreads)}")
        roll_mean = spreads.rolling(window=252).mean()
        roll_std = spreads.rolling(window=252).std()
        print(f"Roll_mean not nan count: {roll_mean.notna().sum()}")
        print(f"Last roll_mean: {roll_mean.iloc[-1]}")
        print(f"Last roll_std: {roll_std.iloc[-1]}")
        # Check if last 252 have na
        last_252 = spreads.tail(252)
        print(f"Last 252 na count: {last_252.isna().sum()}")
else:
    print("Empty or not Series")