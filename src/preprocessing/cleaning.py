import pandas as pd

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Convert dates
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
    df["PreviousTransactionDate"] = pd.to_datetime(df["PreviousTransactionDate"])

    # Strip whitespace from categorical columns
    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].astype(str).str.strip()

    # Remove negative or impossible values
    df = df[df["TransactionAmount"] >= 0]
    df = df[df["CustomerAge"] >= 0]

    return df
