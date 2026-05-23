import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

class KMeansClusteringPlus:
    def __init__(self, n_clusters=5, init_method="k-means++", random_state=42, n_init=10):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.random_state = random_state
        self.n_init = n_init

        self.model = KMeans(
            n_clusters=n_clusters,
            init=init_method,
            random_state=random_state,
            n_init=n_init
        )

    def fit(self, X):
        self.labels = self.model.fit_predict(X)
        return self.labels

    def evaluate(self, X):
        sil = silhouette_score(X, self.labels)
        db = davies_bouldin_score(X, self.labels)
        ch = calinski_harabasz_score(X, self.labels)

        return {
            "silhouette_score": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch
        }

    def get_centroids(self):
        return self.model.cluster_centers_

    def plot_clusters(self, X_pca, save_path="../../results/kmeans"):
        """
        Plot PCA 2D clusters and save to results/kmeans folder.
        """
        os.makedirs(save_path, exist_ok=True)

        plt.figure(figsize=(8, 6))
        plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=self.labels,
            cmap="viridis",
            s=12,
            alpha=0.8
        )

        plt.title(f"KMeans Clusters (K={self.n_clusters})")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.colorbar(label="Cluster")

        file_name = f"cluster_plot_k{self.n_clusters}.png"
        full_path = os.path.join(save_path, file_name)

        plt.savefig(full_path, dpi=300, bbox_inches="tight")
        plt.close()

        return full_path
