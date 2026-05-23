import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_features(df: pd.DataFrame):
    df = df.copy()

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, scaler
