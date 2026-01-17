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
        
    # Narrative Generation
    # Core Narrative
    stress_desc = "calm" if turbulence_score < 50 else "normal" if turbulence_score < 180 else "elevated" if turbulence_score < 370 else "critical"
    trend_desc = "supporting the rally" if spx_status == "ABOVE" else "fighting the trend"
    div_text = "However, a divergence signal warns of a potential trap." if is_divergence else "Internal structure confirms price action."
    
    core_narrative = f"The market is currently {stress_desc} (Turbulence {turbulence_score:.0f}), with price {trend_desc}. {div_text} Volatility remains {'contained' if vix_value < 20 else 'high'} at {vix_value:.1f}."

    # Context Narrative
    fragility_desc = "highly fragile" if absorption_ratio > 0.8 else "showing contagion risk" if absorption_ratio > 0.75 else "resilient"
    sent_desc = "euphoric" if sentiment_score > 80 else "fearful" if sentiment_score < 20 else "neutral"
    
    context_narrative = f"Under the surface, the system is {fragility_desc} with {absorption_ratio:.0%} asset correlation. Investors are currently {sent_desc} (Score: {sentiment_score:.0f}), indicating {risk_appetite.lower()} risk appetite."

    # Summary Narrative (Detailed & Layman)
    # 1. Status & Trend
    if warning_level == "HEALTHY":
        s1 = "The market is currently showing **healthy** vital signs. Price fluctuations are within normal limits, which usually indicates a stable environment for growth."
    elif warning_level == "DIVERGENCE DETECTED":
        s1 = "We are detecting a **hidden warning sign**. While stock prices are rising, the internal pressure (turbulence) is building up, which often happens before a surprise drop."
    else: # Elevated/Critical
        s1 = f"The market is currently **unstable** (Turbulence {turbulence_score:.0f}). Prices are moving erratically, which is a classic signal of increased risk."

    # 2. Structure (Fragility)
    if absorption_ratio > 0.8:
        s2 = "Crucially, the market's internal structure is **fragile**. Almost all stocks are moving in the same direction at the same time, meaning a drop in one could drag down everything else."
    else:
        s2 = "Internally, the market structure is **resilient**. Different sectors are moving independently, acting as a shock absorber against bad news."

    # 3. Sentiment/Context
    if sentiment_score > 60:
        s3 = "Investor sentiment is currently **optimistic**, which can drive prices higher but also leads to complacency."
    elif sentiment_score < 40:
        s3 = "Investor sentiment is **fearful**, which often creates buying opportunities for patient investors."
    else:
        s3 = "Investor sentiment is currently **neutral**, showing no signs of extreme panic or greed."

    # 4. Layman Action
    if regime == "NORMAL":
        s4 = "For most investors, this is a good time to **stay invested** and follow your long-term plan."
    elif regime == "FRAGILE RALLY":
        s4 = "You should be **cautious**. Consider taking some profits off the table or ensuring you have cash ready for a potential dip."
    elif "DIVERGENCE" in regime:
        s4 = "It is smart to **reduce risk** right now. Don't be fooled by the rising prices; focus on protecting what you have made."
    else: # Crash/Sell-off
        s4 = "The safest move right now is **defense**. Avoid making big new bets until the storm passes."
    
    summary_narrative = f"{s1} {s2} {s3} {s4}"

    return {
        "date": date,
        "warning_level": warning_level,
        "badge_color": badge_color,
        "summary_narrative": summary_narrative,
        "core_narrative": core_narrative,
        "context_narrative": context_narrative,
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
