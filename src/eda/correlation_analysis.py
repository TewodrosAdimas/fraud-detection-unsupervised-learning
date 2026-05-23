import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

def compute_correlations(df: pd.DataFrame):
    """Compute Pearson correlation matrix."""
    return df.corr()

def plot_correlation_heatmap(corr_matrix, save_path=None):
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")

    if save_path:
        plt.savefig(f"{save_path}/correlation_heatmap.png")
    plt.close()
