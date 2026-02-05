def compute_risk(deepfake_prob, virality_prob):
    """
    Combines visual deepfake confidence and social virality risk
    """

    # Weighted fusion (explainable & judge-friendly)
    risk_score = (
        0.6 * deepfake_prob +
        0.4 * virality_prob
    )

    if risk_score >= 0.7:
        level = "HIGH RISK"
    elif risk_score >= 0.4:
        level = "MEDIUM RISK"
    else:
        level = "LOW RISK"

    return risk_score, level
