def quality_checker(df):

    issues = {}

    issues["duplicate_rows"] = df.duplicated().sum()

    missing_percent = df.isnull().mean() * 100
    issues["high_missing_columns"] = df.isnull().sum()[df.isnull().sum() > 0].to_dict()

    constant_cols = df.nunique()[df.nunique() == 1].index.tolist()
    issues["constant_columns"] = constant_cols

    high_card_cols = df.nunique()[df.nunique() > 50].index.tolist()
    issues["high_cardinality_columns"] = high_card_cols

    return issues