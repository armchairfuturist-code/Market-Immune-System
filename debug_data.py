from market_immune_system import MarketImmuneSystem
import pandas as pd
import yfinance as yf

def debug_data_fetch():
    with open("debug_output.txt", "w") as f:
        mis = MarketImmuneSystem()
        f.write("Fetching data...\n")
        
        f.write(f"First 10 tickers: {mis._all_tickers[:10]}\n")
        if "SPY" in mis._all_tickers:
            f.write("SPY is in the ticker list.\n")
        else:
            f.write("CRITICAL: SPY missing from ticker list!\n")
            
        f.write(f"Downloading {len(mis._all_tickers)} tickers...\n")
        data = yf.download(
            mis._all_tickers,
            period=f"{mis.fetch_days}d",
            auto_adjust=True,
            progress=False,
            threads=True
        )
        
        f.write("\nData Columns Info:\n")
        f.write(str(data.columns) + "\n")
        f.write("\nData Head:\n")
        f.write(str(data.head()) + "\n")
        
        if isinstance(data.columns, pd.MultiIndex):
            f.write("\nColumns are MultiIndex\n")
            f.write(f"Levels: {data.columns.levels}\n")
        
        # Try to extract SPY directly
        try:
            spy_col = data.xs('SPY', level=1, axis=1) if isinstance(data.columns, pd.MultiIndex) else data['SPY']
            f.write("\nFound SPY directly via xs/indexing\n")
        except Exception as e:
            f.write(f"\nCould not access SPY directly: {e}\n")

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.get_level_values(0):
                prices = data['Close']
                f.write("Used data['Close'] from MultiIndex\n")
            else:
                prices = data
                f.write("Used data directly (fallback)\n")
        elif 'Close' in data.columns:
            prices = data['Close']
            f.write("Used data['Close']\n")
        else:
            prices = data
            f.write("Used data directly\n")
            
        f.write(f"Downloaded columns: {prices.columns.tolist()[:10]}...\n")
        if 'SPY' in prices.columns:
            f.write("SPY is present in downloaded raw data.\n")
            f.write(f"SPY non-null count: {prices['SPY'].count()} / {len(prices)}\n")
        else:
            f.write("CRITICAL: SPY NOT found in downloaded raw data!\n")

        # 2. Ffill
        prices_ffill = prices.ffill()
        
        # 3. Missing Data Check
        missing_pct = prices_ffill.isnull().sum() / len(prices_ffill)
        f.write("\nMissing Data % (Top 10):\n")
        f.write(str(missing_pct.sort_values(ascending=False).head(10)) + "\n")
        
        if 'SPY' in missing_pct:
            f.write(f"\nSPY Missing %: {missing_pct['SPY']:.2%}\n")
        
        valid_cols = missing_pct[missing_pct <= 0.10].index
        f.write(f"\nColumns kept: {len(valid_cols)} / {len(prices.columns)}\n")
        
        if 'SPY' not in valid_cols:
            f.write("CRITICAL: SPY dropped due to >10% missing data\n")
        else:
            f.write("SPY passed missing data check.\n")
            
        prices_filtered = prices_ffill[valid_cols]
        
        # 4. Dropna
        prices_final = prices_filtered.dropna()
        f.write(f"\nShape before dropna: {prices_filtered.shape}\n")
        f.write(f"Shape after dropna: {prices_final.shape}\n")
        
        if prices_final.empty:
            f.write("CRITICAL: dropna() resulted in empty DataFrame!\n")
        else:
            if 'SPY' in prices_final.columns:
                f.write("SPY survives dropna().\n")
            else:
                f.write("CRITICAL: SPY lost during dropna() (unlikely if column exists)\n")

if __name__ == "__main__":
    debug_data_fetch()
