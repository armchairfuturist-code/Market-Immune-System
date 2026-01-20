import lancedb
import pandas as pd
import numpy as np
import pyarrow as pa
import streamlit as st
from datetime import datetime, timedelta  
  
class VectorEngine:  
    def __init__(self, db_path="./vector_db"):  
        self.db = lancedb.connect(db_path)  
        self.table_name = "volume_profiles"  
        # Define formal schema
        self.schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("price_level", pa.float32()),
            pa.field("volume_weight", pa.float32()),
            pa.field("timestamp", pa.string()),
            pa.field("mean_volume_log", pa.float32()),
            pa.field("regime", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 4))
        ])
  
    def vectorize_volume_profile(self, hourly_df, ticker="SPY", days=90):
        end_date = pd.Timestamp.now(tz='UTC')
        start_date = end_date - timedelta(days=days)
        
        # Access multi-index dataframe correctly
        try:
            df = hourly_df.xs(ticker, axis=1, level=1).loc[start_date:end_date]
        except KeyError:
            return

        if df.empty:
            return

        profiles = []  
        for i, row in df.iterrows():  
            price_range = np.linspace(row['Low'], row['High'], 10)  
            vol_per_level = row['Volume'] / len(price_range)  
            for p in price_range:  
                profiles.append({  
                    "price_level": p,  
                    "volume_weight": vol_per_level,  
                    "timestamp": str(i),  
                    "mean_volume_log": np.log(vol_per_level + 1)  
                })  
        
        profile_df = pd.DataFrame(profiles)  
        agg_df = profile_df.groupby('price_level').agg({  
            'volume_weight': 'sum',  
            'mean_volume_log': 'mean'  
        }).reset_index()  
        
        vectors = []  
        for _, row in agg_df.iterrows():  
            vector = [  
                float(row['price_level']),  
                float(row['volume_weight']),  
                datetime.now().timestamp(),  
                float(row['mean_volume_log'])  
            ]  
            vectors.append({  
                "id": f"{ticker}_{row['price_level']:.0f}",  
                "price_level": row['price_level'],  
                "volume_weight": row['volume_weight'],  
                "timestamp": str(datetime.now()),  
                "mean_volume_log": row['mean_volume_log'],  
                "regime": "neutral",  
                "vector": vector  
            })  
        
        # Check existence before opening
        if self.table_name not in self.db.table_names():
            self.db.create_table(self.table_name, data=vectors, schema=self.schema)
        else:
            table = self.db.open_table(self.table_name)
            table.add(vectors)  
  
    def get_vpoc_level(self, current_price, regime="neutral"):  
        if self.table_name not in self.db.table_names():
            return current_price
            
        table = self.db.open_table(self.table_name)  
        results = table.search().limit(1000).to_pandas()  
        
        if results.empty:  
            return current_price  
            
        agg = results.groupby('price_level')['volume_weight'].sum()  
        vpoc = agg.idxmax()
        return float(vpoc) # Ensure return
