import matplotlib.pyplot as plt
import seaborn as sns
import os


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def plot_transactions_per_account(df, save_path=None):
    account_counts = df["AccountID"].value_counts().sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x=account_counts.index, y=account_counts.values, color="steelblue")

    plt.title("Number of Transactions per AccountID")
    plt.xlabel("AccountID")
    plt.ylabel("Transaction Count")
    plt.xticks([], [])  # Hide labels if too many accounts

    if save_path:
        ensure_dir(save_path)
        plt.savefig(f"{save_path}/transactions_per_account.png")

    plt.close()


def plot_histograms(df, numeric_cols, save_path=None):
    ensure_dir(save_path)
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[col], kde=True)
        plt.title(f"Histogram — {col}")
        if save_path:
            plt.savefig(f"{save_path}/{col}_hist.png")
        plt.close()


def plot_boxplots(df, numeric_cols, save_path=None):
    ensure_dir(save_path)
    for col in numeric_cols:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        plt.title(f"Boxplot — {col}")
        if save_path:
            plt.savefig(f"{save_path}/{col}_box.png")
        plt.close()


def plot_categorical_counts(df, categorical_cols, save_path=None):
    ensure_dir(save_path)
    for col in categorical_cols:
        plt.figure(figsize=(6, 4))
        df[col].value_counts().plot(kind="bar")
        plt.title(f"Category Counts — {col}")
        if save_path:
            plt.savefig(f"{save_path}/{col}_counts.png")
        plt.close()


def plot_time_series(df, time_col, value_col, save_path=None):
    ensure_dir(save_path)
    plt.figure(figsize=(10, 5))
    plt.plot(df[time_col], df[value_col])
    plt.title(f"Time Series — {value_col} over time")
    if save_path:
        plt.savefig(f"{save_path}/{value_col}_timeseries.png")
    plt.close()
