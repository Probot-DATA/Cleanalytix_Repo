import pandas as pd


def infer_and_fix_types(df):
    """Stabilize obvious numeric, datetime, and boolean columns before profiling."""
    df = df.copy()

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            df[col] = pd.Series(series.astype("category").values, index=df.index)
            continue

        # Skip if already numeric or datetime
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            df[col] = pd.Series(series.values, index=df.index)
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            df[col] = pd.Series(series.values, index=df.index)
            continue

        # Try numeric conversion
        numeric_conv = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric_conv.notna().mean()

        if numeric_ratio > 0.9:
            df[col] = pd.Series(numeric_conv.values, index=df.index)
            continue

        # Try datetime conversion
        datetime_conv = pd.to_datetime(series, errors="coerce", format="mixed")
        datetime_ratio = datetime_conv.notna().mean()

        if datetime_ratio > 0.9:
            df[col] = pd.Series(datetime_conv.values, index=df.index)
            continue

        # Fallback: keep the original values and index shape intact.
        df[col] = pd.Series(series.values, index=df.index)

    return df
