import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """

    df = pd.read_csv(path)

    return df