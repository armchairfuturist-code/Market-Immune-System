def get_cycle_playbook(phase_name):
    """
    Returns the Layman's Playbook and Rotation Strategy for the given cycle phase.
    """
    
    # Normalize phase name (e.g. "Phase I: Early Cycle..." -> "EARLY")
    phase_key = "UNKNOWN"
    if "Early" in phase_name: phase_key = "EARLY"
    elif "Mid" in phase_name: phase_key = "MID"
    elif "Late" in phase_name: phase_key = "LATE"
    elif "Recession" in phase_name: phase_key = "RECESSION"
    
    playbooks = {
        "EARLY": {
            "title": "🌱 Early Cycle (Recovery)",
            "context": "Economy is waking up. Interest rates are low, credit is cheap. Risk-on.",
            "style": "Value & Small Caps",
            "capital_rotation": {
                "Buy": ["Financials (XLF)", "Real Estate (XLRE)", "Consumer Discretionary (XLY)", "Industrials (XLI)"],
                "Sell/Avoid": ["Utilities (XLU)", "Consumer Staples (XLP)"]
            },
            "layman_strategy": "Banks lend more, people buy houses and cars. Buy the companies that build and finance things."
        },
        "MID": {
            "title": "🚀 Mid Cycle (Expansion)",
            "context": "Peak growth. Corporate profits soar. The 'Goldilocks' zone.",
            "style": "Growth & Momentum",
            "capital_rotation": {
                "Buy": ["Technology (XLK)", "Communication Services (XLC)"],
                "Sell/Avoid": ["Utilities (XLU)", "Materials (XLB)"]
            },
            "layman_strategy": "Things are good. Bet on innovation and tech giants driving the future."
        },
        "LATE": {
            "title": "🔥 Late Cycle (Overheating)",
            "context": "Inflation rises, Fed hikes rates to cool things down. Volatility returns.",
            "style": "Quality & Low Volatility",
            "capital_rotation": {
                "Buy": ["Energy (XLE)", "Materials (XLB)", "Healthcare (XLV)"],
                "Sell/Avoid": ["Technology (XLK)", "Consumer Discretionary (XLY)"]
            },
            "layman_strategy": "Prices are rising (Inflation). Buy commodities (Oil/Copper) and essential medicine."
        },
        "RECESSION": {
            "title": "🛑 Recession (Contraction)",
            "context": "Economy shrinks. Profits fall. Cash is King.",
            "style": "Defensive & Yield",
            "capital_rotation": {
                "Buy": ["Consumer Staples (XLP)", "Utilities (XLU)", "Govt Bonds (TLT)"],
                "Sell/Avoid": ["Financials (XLF)", "Industrials (XLI)", "Real Estate (XLRE)"]
            },
            "layman_strategy": "Hunker down. People still need toothpaste, electricity, and dividends. Safety first."
        }
    }
    
    return playbooks.get(phase_key, {
        "title": "Unknown Phase",
        "context": "Insufficient data to determine cycle.",
        "style": "Neutral",
        "capital_rotation": {"Buy": [], "Sell/Avoid": []},
        "layman_strategy": "Stay diversified."
    })
