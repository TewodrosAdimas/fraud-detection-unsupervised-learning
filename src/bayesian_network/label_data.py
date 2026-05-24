import pandas as pd

def label_dataset(raw_path, anomaly_path, output_path):

    # Load datasets
    df_raw = pd.read_csv(raw_path)
    df_anom = pd.read_csv(anomaly_path)

    # Default all rows to normal
    df_raw["anomaly"] = 0

    # DBSCAN anomalies are cluster == -1
    anomaly_indices = df_anom[
        df_anom["cluster"] == -1
    ]["index"].tolist()

    # Label anomalies
    df_raw.loc[anomaly_indices, "anomaly"] = 1

    # Save
    df_raw.to_csv(output_path, index=False)

    return df_raw