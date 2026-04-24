def reliability_score(profile, quality_issues):

    score = 100

    # Duplicate penalty
    score -= quality_issues["duplicate_rows"] * 0.1

    # Missing values penalty
    missing_penalty = sum(quality_issues["high_missing_columns"].values()) / 100
    score -= missing_penalty * 10   

    # Constant columns penalty
    score -= len(quality_issues["constant_columns"]) * 5

    # High cardinality penalty
    score -= len(quality_issues["high_cardinality_columns"]) * 2

    # Keep score in bounds
    score = max(0, min(100, score))

    return round(score, 2)