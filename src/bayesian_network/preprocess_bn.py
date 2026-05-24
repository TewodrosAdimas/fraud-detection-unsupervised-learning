import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def discretize_dataset(df, n_bins=4):
    """
    Discretize ONLY numeric features for Bayesian Network structure learning.
    Automatically detects numeric columns and avoids all object/string columns.
    """

    df_disc = df.copy()

    # Automatically detect numeric columns
    numeric_cols = df_disc.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Remove anomaly label if present
    if "anomaly" in numeric_cols:
        numeric_cols.remove("anomaly")

    print("Numeric columns being discretized:", numeric_cols)

    # If no numeric columns, return as-is
    if len(numeric_cols) == 0:
        return df_disc

    # Discretize numeric columns only
    disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
    df_disc[numeric_cols] = disc.fit_transform(df_disc[numeric_cols])

    return df_disc
