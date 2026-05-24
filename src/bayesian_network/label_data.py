import pandas as pd

def label_dataset(raw_path, anomaly_path, output_path):
    """
    Merge raw dataset with DBSCAN anomaly labels and save processed file.
    """

    # Load raw dataset
    df_raw = pd.read_csv(raw_path)

    # Load anomaly table (must contain index + anomaly label)
    df_anom = pd.read_csv(anomaly_path)

    # Ensure index column exists
    if "index" not in df_anom.columns:
        raise ValueError("Anomaly table must contain an 'index' column.")

    # Set index for merging
    df_anom = df_anom.set_index("index")

    # Create anomaly column (1 = anomaly, 0 = normal)
    df_anom["anomaly"] = (df_anom["cluster"] == "noise").astype(int)

    # Merge with raw dataset
    df_raw_labeled = df_raw.copy()
    df_raw_labeled["anomaly"] = df_anom["anomaly"]

    # Save processed dataset
    df_raw_labeled.to_csv(output_path, index=False)

    return df_raw_labeled
