import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

class HierarchicalClustering:
    def __init__(self, n_clusters=3, linkage_method="ward", save_dir="../results/hierarchical"):
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.labels_ = None
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, X):
        model = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage_method
        )
        self.labels_ = model.fit_predict(X)
        return self.labels_

    def plot_clusters(self, X):
        if self.labels_ is None:
            raise ValueError("Model must be fitted before plotting.")

        filename = f"hierarchical_{self.linkage_method}_k{self.n_clusters}.png"
        save_path = os.path.join(self.save_dir, filename)

        plt.figure(figsize=(8,6))
        plt.scatter(X[:,0], X[:,1], c=self.labels_, cmap="tab10", s=20)
        plt.title(f"Hierarchical Clustering ({self.linkage_method}, k={self.n_clusters})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Cluster plot saved to: {save_path}")

    def plot_dendrogram(self, X):
        Z = linkage(X, method=self.linkage_method)

        filename = f"dendrogram_{self.linkage_method}.png"
        save_path = os.path.join(self.save_dir, filename)

        plt.figure(figsize=(10, 6))
        dendrogram(Z, truncate_mode="level", p=5)
        plt.title(f"Dendrogram ({self.linkage_method})")
        plt.xlabel("Samples")
        plt.ylabel("Distance")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Dendrogram saved to: {save_path}")
