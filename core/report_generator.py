import pandas as pd
import datetime

def generate_immune_report(
    date,
    turbulence_score,
    spx_price,
    spx_ma50,
    vix_value,
    is_divergence,
    regime,
    sentiment_score,
    absorption_ratio,
    days_elevated=0
):
    """
    Generates a structured data dictionary for the Market Immune System report.
    Returns: dict
    """
    
    # 1. Warning Level
    warning_level = "HEALTHY"
    badge_color = "green"
    
    if turbulence_score > 370:
        warning_level = "CRITICAL"
        badge_color = "red"
    elif turbulence_score > 180:
        warning_level = "ELEVATED"
        badge_color = "yellow"
    elif is_divergence:
        warning_level = "DIVERGENCE DETECTED"
        badge_color = "yellow"
        
    # 2. SPX Status
    spx_status = "ABOVE" if spx_price > spx_ma50 else "BELOW"
    
    # 3. Correlation Breakdown
    corr_breakdown = "Yes" if absorption_ratio > 0.85 else "No"
    
    # 4. Risk Appetite
    risk_appetite = "High"
    if regime in ["SYSTEMIC SELL-OFF", "CRASH ALERT"]:
        risk_appetite = "Low"
    elif regime in ["STRUCTURAL DIVERGENCE", "FRAGILE RALLY"]:
        risk_appetite = "Medium"
        
    # 5. Interpretation
    interpretation = []
    if warning_level == "HEALTHY":
        interpretation.append("Market showing normal stress levels.")
        interpretation.append("No concerning correlation breakdowns.")
        interpretation.append("Risk-on environment.")
    elif warning_level == "DIVERGENCE DETECTED":
        interpretation.append("Price rising but internal structure weakening.")
        interpretation.append("Volatility rising under the surface.")
        interpretation.append("Trap potential high.")
    else:
        interpretation.append(f"Market in {regime} state.")
        interpretation.append("Defensive positioning required.")
        
    # 6. Actionable Insights
    actions = []
    if regime == "NORMAL":
        actions.append("Maintain equity exposure. Standard stops.")
    elif regime == "FRAGILE RALLY":
        actions.append("Tighten stops. Do not chase new highs.")
    elif regime == "STRUCTURAL DIVERGENCE":
        actions.append("Reduce leverage. Hedge long positions.")
    elif regime == "SYSTEMIC SELL-OFF":
        actions.append("Move to Cash/Treasuries. Avoid 'buying the dip'.")
    elif regime == "CRASH ALERT":
        actions.append("Maximum defense. Liquidate speculative assets.")
        
    return {
        "date": date,
        "warning_level": warning_level,
        "badge_color": badge_color,
        "core_metrics": {
            "Market Turbulence": f"{turbulence_score:.1f}",
            "Days Elevated": days_elevated,
            "SPX Level": f"{spx_price:.2f} ({spx_status} MA)",
            "VIX Level": f"{vix_value:.2f}",
            "Divergence": "YES" if is_divergence else "NO"
        },
        "context_metrics": {
            "Correlation Break": f"{corr_breakdown} ({absorption_ratio:.0%})",
            "Sentiment": f"{sentiment_score:.0f}/100",
            "Risk Appetite": risk_appetite
        },
        "interpretation": interpretation,
        "actions": actions
    }
