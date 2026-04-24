def data_profiler(df):
    profile = {
        "shape": df.shape,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "numeric_summary": df.describe().to_dict(),
        "columns": df.columns.tolist()
    }

    return profile