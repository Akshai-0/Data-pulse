def reliability_score(profile, quality_issues, outliers, clusters):

    score = 100

    # 🔴 Missing values (moderate impact)
    missing_penalty = sum(quality_issues["high_missing_columns"].values())
    score -= missing_penalty * 1.2

    # 🔴 Duplicates (low impact)
    score -= quality_issues["duplicate_rows"] * 0.3

    # 🟡 Constant columns (medium impact)
    score -= len(quality_issues["constant_columns"]) * 4

    # 🔴 Outliers (important)
    total_outliers = sum(outliers.values())
    score -= total_outliers * 0.5

    # 🔴 Cluster anomalies (strong impact)
    score -= len(clusters["small_clusters"]) * 5

    # Clamp
    score = max(0, min(100, score))

    return round(score, 2)
