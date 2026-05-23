import pandas as pd
import numpy as np

def summarize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataset summary: dtypes, missing %, unique counts."""
    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing_pct": df.isna().mean() * 100,
        "unique_values": df.nunique()
    })
    return summary.reset_index().rename(columns={"index": "column"})

def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing values count and percentage."""
    missing = df.isna().sum().reset_index()
    missing.columns = ["column", "missing_count"]
    missing["missing_pct"] = (missing["missing_count"] / len(df)) * 100
    return missing

def unique_values_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return number of unique values per column."""
    return df.nunique().reset_index().rename(columns={"index": "column", 0: "unique_count"})

def detect_outliers_iqr(df: pd.DataFrame, numeric_cols):
    """Detect outliers using IQR method."""
    outlier_info = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col].count()

        outlier_info.append({
            "column": col,
            "outliers": outliers,
            "lower_bound": lower,
            "upper_bound": upper
        })

    return pd.DataFrame(outlier_info)
