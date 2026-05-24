import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def discretize_dataset(df, n_bins=4, strategy="quantile", target_col="anomaly"):
    """
    Discretize numeric features for Bayesian Network structure learning.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    n_bins : int
        Number of bins for discretization
    strategy : str
        'quantile', 'uniform', or 'kmeans'
    target_col : str
        Target column to exclude from discretization
    
    Returns:
    --------
    pd.DataFrame
        Discretized dataset
    """

    df_disc = df.copy()

    # Separate target
    target = None
    if target_col in df_disc.columns:
        target = df_disc[target_col]
        df_disc = df_disc.drop(columns=[target_col])

    # Detect numeric columns only
    numeric_cols = df_disc.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Keep boolean columns as-is
    bool_cols = df_disc.select_dtypes(include=["bool"]).columns.tolist()

    # Apply discretization only to numeric columns
    if len(numeric_cols) > 0:
        discretizer = KBinsDiscretizer(
            n_bins=n_bins,
            encode="ordinal",
            strategy=strategy
        )

        df_disc[numeric_cols] = discretizer.fit_transform(df_disc[numeric_cols])

    # Reattach target
    if target is not None:
        df_disc[target_col] = target.values

    return df_disc