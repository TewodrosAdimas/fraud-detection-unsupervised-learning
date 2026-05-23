import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows.
    """

    return df.drop_duplicates()


def missing_values_summary(df: pd.DataFrame) -> pd.Series:
    """
    Return missing values count.
    """

    return df.isnull().sum()


def numerical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return summary statistics for numerical columns.
    """

    return df.describe()


def categorical_columns(df: pd.DataFrame):
    """
    Return categorical columns.
    """

    return df.select_dtypes(include=['object']).columns