import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Time since previous transaction
    df["TimeSincePrev"] = (
        df["TransactionDate"] - df["PreviousTransactionDate"]
    ).dt.total_seconds().fillna(0)

    # Amount-to-balance ratio
    df["AmountBalanceRatio"] = df["TransactionAmount"] / (df["AccountBalance"] + 1)

    # Login attempt risk
    df["LoginRisk"] = df["LoginAttempts"].apply(lambda x: 1 if x > 1 else 0)

    # Transaction velocity (per account)
    df["TransactionCount"] = df.groupby("AccountID").cumcount() + 1

    return df
