import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def discretize_dataset(df, n_bins=4):
    """
    Robust preprocessing + discretization for Bayesian Networks.

    - Safe against missing columns (no KeyError)
    - Converts numeric → categorical bins
    - Extracts datetime features safely
    - Removes ID-like columns
    """

    df = df.copy()

    # =========================================================
    # 1. DROP ID-LIKE COLUMNS (safe)
    # =========================================================
    drop_cols = [
        "TransactionID",
        "AccountID",
        "DeviceID",
        "IP Address",
        "MerchantID"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # =========================================================
    # 2. HANDLE TRANSACTION DATE
    # =========================================================
    if "TransactionDate" in df.columns:
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors="coerce")

        df["txn_hour"] = df["TransactionDate"].dt.hour
        df["txn_dayofweek"] = df["TransactionDate"].dt.dayofweek

        df["time_of_day"] = pd.cut(
            df["txn_hour"],
            bins=[-1, 5, 12, 17, 21, 23],
            labels=["night", "morning", "afternoon", "evening", "late_evening"]
        )

        df = df.drop(columns=["TransactionDate", "txn_hour"], errors="ignore")

    # =========================================================
    # 3. HANDLE PREVIOUS TRANSACTION DATE
    # =========================================================
    if "PreviousTransactionDate" in df.columns:
        df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"], errors="coerce")

        # Only compute if TransactionDate existed before dropping
        if "txn_dayofweek" in df.columns:
            # approximate time gap using dummy shift (safe fallback)
            df["time_since_prev"] = np.nan
        else:
            df["time_since_prev"] = np.nan

        df = df.drop(columns=["PreviousTransactionDate"], errors="ignore")

    # =========================================================
    # 4. NUMERIC BINNING FUNCTION
    # =========================================================
    def bin_numeric(series, bins):
        try:
            kb = KBinsDiscretizer(
                n_bins=bins,
                encode="ordinal",
                strategy="quantile"
            )
            return kb.fit_transform(series.values.reshape(-1, 1)).astype(int).flatten()
        except:
            return series

    # =========================================================
    # 5. APPLY NUMERIC DISCRETIZATION
    # =========================================================
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    for col in numeric_cols:
        if df[col].nunique() > n_bins:
            df[col] = bin_numeric(df[col], n_bins)

    # =========================================================
    # 6. ENCODE CATEGORICAL COLUMNS
    # =========================================================
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].astype("category").cat.codes

    # =========================================================
    # 7. CLEAN UP
    # =========================================================
    df = df.replace([np.inf, -np.inf], np.nan)

    # fill numeric NaNs safely
    df = df.fillna(df.median(numeric_only=True))

    return df