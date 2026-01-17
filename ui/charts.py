import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import config

def create_gauge_chart(value, title, min_val, max_val, thresholds, inverse=False):
    """
    Creates a Gauge Chart for metrics like Turbulence or Fragility.
    thresholds: dict {limit: color} e.g. {180: "orange", 370: "red"}
    inverse: If True, Low is Bad (Red), High is Good (Green). 
             Default False (Low is Good/Green).
    """
    
    # Determine bar color based on value
    bar_color = config.COLOR_ACCENT_GREEN
    
    sorted_thresh = sorted(thresholds.keys())
    
    # Logic for "Low is Good" (Standard)
    # < T1 = Green, > T1 = Orange, > T2 = Red
    if not inverse:
        if value > sorted_thresh[-1]:
            bar_color = config.COLOR_ACCENT_RED
        elif value > sorted_thresh[0]:
            bar_color = config.COLOR_ACCENT_AMBER
    else:
        # Logic for "High is Good" (e.g. Liquidity?) - Not currently used but good to have
        pass

    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 14, 'color': "#888"}},
        number = {'font': {'color': "white"}},
        gauge = {
            'axis': {'range': [min_val, max_val], 'tickwidth': 1, 'tickcolor': "#333"},
            'bar': {'color': bar_color},
            'bgcolor': "#1E1E2E",
            'borderwidth': 0,
            'steps': [
                {'range': [min_val, sorted_thresh[0]], 'color': "rgba(0, 200, 83, 0.1)"},
                {'range': [sorted_thresh[0], sorted_thresh[1]], 'color': "rgba(255, 171, 0, 0.1)"},
                {'range': [sorted_thresh[1], max_val], 'color': "rgba(255, 23, 68, 0.1)"}
            ],
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"},
        height=150,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    
    return fig

def plot_divergence_chart(prices, turbulence, ma_window=50, futures_data=None):
    """
    Creates the main Divergence Detector chart.
    Turbulence (Left Axis), Price (Right Axis).
    Includes Thresholds, MA, and Futures Projection.
    """
    
    # Calculate MA
    ma = prices.rolling(window=ma_window).mean()
    
    # Align dates
    common_index = prices.index.intersection(turbulence.index)
    prices = prices.loc[common_index]
    turbulence = turbulence.loc[common_index]
    ma = ma.loc[common_index]
    
    # Calculate Dynamic Thresholds (Data Parity)
    p95 = turbulence.quantile(0.95)
    p99 = turbulence.quantile(0.99)
    
    # Identify Green Shading Zones (Divergence Signal)
    # Logic: Turbulence > 95th Percentile AND Price > 50MA
    mask = (turbulence > config.REGIME_DIVERGENCE_THRESHOLD) & (prices > ma)
    
    # Create Figure with Secondary Y Axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Turbulence (Area Chart) - Primary Axis (Left)
    fig.add_trace(
        go.Scatter(
            x=turbulence.index, 
            y=turbulence, 
            name="Market Turbulence",
            fill='tozeroy',
            line=dict(color=config.COLOR_ACCENT_RED, width=1.5),
            opacity=0.8
        ),
        secondary_y=False
    )
    
    # Add SPX Price (Line) - Secondary Axis (Right)
    fig.add_trace(
        go.Scatter(
            x=prices.index, 
            y=prices, 
            name="SPX",
            line=dict(color="#ECF0F1", width=2)
        ),
        secondary_y=True
    )
    
    # Add SPX MA (Dotted Line) - Secondary Axis (Right)
    fig.add_trace(
        go.Scatter(
            x=ma.index, 
            y=ma, 
            name=f"SPX {ma_window}-MA",
            line=dict(color="#AAB7B8", width=1.5, dash="dash")
        ),
        secondary_y=True
    )
    
    # Futures Projection
    if futures_data is not None and not futures_data.empty and not prices.empty:
        try:
            recent_futures = futures_data.tail(5)
            if len(recent_futures) > 1:
                f_start = recent_futures.iloc[0]
                f_end = recent_futures.iloc[-1]
                pct_change = (f_end - f_start) / f_start
                daily_drift = pct_change / len(recent_futures)
                
                last_price = prices.iloc[-1]
                last_date = prices.index[-1]
                
                proj_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 15)]
                proj_prices = [last_price * (1 + daily_drift * i) for i in range(1, 15)]
                
                fig.add_trace(
                    go.Scatter(
                        x=proj_dates,
                        y=proj_prices,
                        name="Trend Projection",
                        line=dict(color="#00C853" if daily_drift > 0 else "#FF1744", width=2, dash="dot")
                    ),
                    secondary_y=True
                )
        except Exception as e:
            print(f"Projection Error: {e}")

    # Add Threshold Lines (Left Axis) - Reference Parity
    fig.add_hline(
        y=config.REGIME_DIVERGENCE_THRESHOLD, 
        line_dash="dash", 
        line_color="#F1C40F", # Gold
        annotation_text=f"Trap Zone ({config.REGIME_DIVERGENCE_THRESHOLD})", 
        annotation_position="top left",
        secondary_y=False
    )
    fig.add_hline(
        y=config.REGIME_TURBULENCE_CRASH, 
        line_dash="dash", 
        line_color="#E74C3C", # Red
        annotation_text=f"CRASH Level ({config.REGIME_TURBULENCE_CRASH})", 
        annotation_position="top left",
        secondary_y=False
    )
    
    # Add Green Shading (VRects)
    is_active = False
    start_date = None
    
    for date, val in mask.items():
        if val and not is_active:
            is_active = True
            start_date = date
        elif not val and is_active:
            is_active = False
            end_date = date
            
            fig.add_vrect(
                x0=start_date, x1=end_date,
                fillcolor=config.COLOR_ACCENT_GREEN, opacity=0.15,
                layer="below", line_width=0,
            )
            
    if is_active:
        fig.add_vrect(
            x0=start_date, x1=mask.index[-1],
            fillcolor=config.COLOR_ACCENT_GREEN, opacity=0.15,
            layer="below", line_width=0,
        )

    # Layout Updates
    fig.update_layout(
        title="<b>DIVERGENCE DETECTOR: Turbulence vs SPX</b>",
        template="plotly_dark",
        paper_bgcolor=config.COLOR_BG,
        plot_bgcolor=config.COLOR_BG,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    
    # Left Axis: Turbulence
    fig.update_yaxes(title_text="Market Turbulence", secondary_y=False, showgrid=True, gridcolor="#333", range=[0, 650])
    # Right Axis: Price
    fig.update_yaxes(title_text="SPX Level", secondary_y=True, showgrid=False)
    
    return fig

def plot_narrative_battle(crypto_df, ai_df):
    """
    Plots Relative Strength of Crypto vs AI.
    """
    # Normalize to start = 100
    if crypto_df.empty or ai_df.empty:
        return go.Figure()

    c_norm = (crypto_df / crypto_df.iloc[0]) * 100
    a_norm = (ai_df / ai_df.iloc[0]) * 100
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=c_norm.index, y=c_norm, name="Crypto (Basket)", line=dict(color="#F7931A")))
    fig.add_trace(go.Scatter(x=a_norm.index, y=a_norm, name="AI (Basket)", line=dict(color="#00A4E3")))
    
    fig.update_layout(
        title="Narrative Battle: Crypto vs. AI",
        template="plotly_dark",
        paper_bgcolor=config.COLOR_CARD,
        plot_bgcolor=config.COLOR_CARD,
        height=300
    )
    return fig