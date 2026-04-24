def outlier_detection(df):

    numeric_cols = df.select_dtypes(include=['int64', 'float64'])

    outlier_counts = {}

    for col in numeric_cols.columns:
        Q1 = numeric_cols[col].quantile(0.25)
        Q3 = numeric_cols[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = numeric_cols[(numeric_cols[col] < lower) | (numeric_cols[col] > upper)]

        outlier_counts[col] = len(outliers)

    return outlier_counts