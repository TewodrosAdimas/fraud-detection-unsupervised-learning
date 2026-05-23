import pandas as pd

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # One-hot encode small categories
    small_cats = ["TransactionType", "Channel", "CustomerOccupation"]
    df = pd.get_dummies(df, columns=small_cats, drop_first=True)

    # Frequency encode high-cardinality categories
    high_cats = ["AccountID", "DeviceID", "IP Address", "MerchantID", "Location"]

    for col in high_cats:
        freq = df[col].value_counts()
        df[col + "_freq"] = df[col].map(freq)

    df.drop(columns=high_cats, inplace=True)

    return df
