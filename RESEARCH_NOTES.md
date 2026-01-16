# Market Immune System - Research Notes

## Future Integration Opportunities

This document outlines potential data sources for future enhancements to the Market Immune System dashboard.

---

## 1. Dark Pool Index (DIX) Integration

### Overview
Dark Pools are private exchanges where institutional investors trade large blocks without affecting public market prices. Tracking dark pool activity can provide early signals of institutional positioning.

### Data Source: SqueezeMetrics
- **Website**: https://squeezemetrics.com/monitor/dix
- **Data**: DIX (Dark Index) and GEX (Gamma Exposure)
- **Signal Logic**: 
  - When DIX > 50%, institutions are likely accumulating (bullish)
  - When DIX < 40%, institutions may be distributing (bearish)
  - DIX spikes often precede market reversals by 1-3 days

### Implementation Notes
```python
# Potential implementation approach
# SqueezeMetrics publishes daily CSV files

def get_dark_pool_index():
    """
    Fetch DIX data from SqueezeMetrics.
    Note: May require manual download or paid API subscription.
    """
    # Option 1: Manual CSV download from website
    # Option 2: Check if they have a public API
    # Option 3: Use web scraping (check ToS first)
    pass
```

### Challenges
- SqueezeMetrics may require subscription for API access
- Data updates daily after market close
- Historical data limited on free tier

---

## 2. Senate Trading (Quiver Quantitative)

### Overview
US Senators are required to disclose their trades within 45 days. Research shows their trades often outperform the market, suggesting access to non-public information.

### Data Source: Quiver Quantitative
- **Website**: https://www.quiverquant.com/
- **API**: https://api.quiverquant.com/
- **Available Data**:
  - Senate trades (disclosed filings)
  - House trades
  - Congress member performance tracking

### Signal Logic
```python
# Proposed implementation

def get_senate_trading_signals():
    """
    Fetch recent Senate trading activity.
    
    Signals:
    - Net selling in Tech sector = potential warning
    - Concentrated buying in specific sector = opportunity
    - Unusual volume in single stock = potential catalyst
    """
    # API endpoint: GET https://api.quiverquant.com/beta/live/senatetrading
    # Requires API key (free tier available)
    pass
```

### Proposed Widget: Insider Heatmap
Display a simple heatmap showing:
- Sectors with most Senate buying (green)
- Sectors with most Senate selling (red)
- Time period: Last 30 days

### Implementation Priority
**Medium** - Free API available, relatively straightforward to implement

---

## 3. Additional Data Sources to Research

### 3.1 Options Flow (Unusual Activity)
- **Source**: Unusual Whales, FlowAlgo
- **Signal**: Large options purchases often precede moves
- **Challenge**: Most services are paid subscriptions

### 3.2 Fed Funds Futures
- **Source**: CME Group
- **Signal**: Market expectations for rate changes
- **Implementation**: Calculate implied probability of rate hike/cut

### 3.3 Credit Default Swaps (CDS)
- **Source**: IHS Markit, Bloomberg
- **Signal**: Rising CDS spreads indicate credit stress
- **Challenge**: Expensive data subscriptions required

### 3.4 Retail Sentiment
- **Source**: Reddit, StockTwits, Twitter API
- **Signal**: Extreme retail bullishness often marks local tops
- **Implementation**: NLP on social media posts

---

## Implementation Roadmap

| Priority | Feature | Data Source | Estimated Effort |
|----------|---------|-------------|------------------|
| High | Senate Trading Widget | Quiver Quantitative | 1-2 days |
| Medium | Dark Pool DIX | SqueezeMetrics | 2-3 days |
| Low | Options Flow | Unusual Whales | 3-5 days |
| Low | Fed Funds Implied | CME | 1 day |

---

## Notes

- All external data sources should be wrapped in try/except blocks
- Cache external API calls appropriately (1-24 hours depending on data freshness needs)
- Respect rate limits and Terms of Service for each API
- Consider fallback displays when data is unavailable
