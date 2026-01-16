import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import config

def plot_divergence_chart(prices, turbulence, ma_window=50):
    """
    Creates the main Divergence Detector chart.
    Turbulence (Left Axis), Price (Right Axis).
    Includes Thresholds and MA.
    """
    
    # Calculate MA
    ma = prices.rolling(window=ma_window).mean()
    
    # Identify Green Shading Zones
    # "Turbulence > 180 AND SPX > 50-day MA"
    
    # Align dates
    common_index = prices.index.intersection(turbulence.index)
    prices = prices.loc[common_index]
    turbulence = turbulence.loc[common_index]
    ma = ma.loc[common_index]
    
    # Calculate Thresholds
    p95 = turbulence.quantile(0.95)
    p99 = turbulence.quantile(0.99)
    
    # Create mask
    mask = (turbulence > config.REGIME_TURBULENCE_HIGH) & (prices > ma)
    
    # Create Figure with Secondary Y Axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add Turbulence (Area Chart) - Primary Axis (Left)
    fig.add_trace(
        go.Scatter(
            x=turbulence.index, 
            y=turbulence, 
            name="Turbulence",
            fill='tozeroy',
            line=dict(color=config.COLOR_ACCENT_RED, width=1),
            opacity=0.5
        ),
        secondary_y=False
    )
    
    # Add SPX Price (Line) - Secondary Axis (Right)
    fig.add_trace(
        go.Scatter(
            x=prices.index, 
            y=prices, 
            name="SPX Price",
            line=dict(color="#FFFFFF", width=2)
        ),
        secondary_y=True
    )
    
    # Add SPX MA (Dotted Line) - Secondary Axis (Right)
    fig.add_trace(
        go.Scatter(
            x=ma.index, 
            y=ma, 
            name=f"SPX {ma_window}d MA",
            line=dict(color="#CCCCCC", width=1, dash="dot")
        ),
        secondary_y=True
    )
    
    # Add Threshold Lines (Left Axis)
    fig.add_hline(y=p95, line_dash="dot", line_color=config.COLOR_ACCENT_AMBER, annotation_text="95% Warning", secondary_y=False)
    fig.add_hline(y=p99, line_dash="dot", line_color=config.COLOR_ACCENT_RED, annotation_text="99% Extreme", secondary_y=False)
    
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
            
            # Add shape
            fig.add_vrect(
                x0=start_date, x1=end_date,
                fillcolor=config.COLOR_ACCENT_GREEN, opacity=0.1,
                layer="below", line_width=0,
            )
            
    if is_active:
        fig.add_vrect(
            x0=start_date, x1=mask.index[-1],
            fillcolor=config.COLOR_ACCENT_GREEN, opacity=0.1,
            layer="below", line_width=0,
        )

    # Layout Updates
    fig.update_layout(
        title="Divergence Detector (Turbulence vs. Price)",
        template="plotly_dark",
        paper_bgcolor=config.COLOR_BG,
        plot_bgcolor=config.COLOR_BG,
        height=600,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center")
    )
    
    # Left Axis: Turbulence
    fig.update_yaxes(title_text="Turbulence Score", secondary_y=False, showgrid=True, gridcolor="#333", range=[0, 1000])
    # Right Axis: Price
    fig.update_yaxes(title_text="SPX Price", secondary_y=True, showgrid=False)
    
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
