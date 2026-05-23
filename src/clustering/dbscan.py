import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class DBSCANClustering:
    def __init__(self, eps=0.5, min_samples=5, save_dir="../results/dbscan"):
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, X):
        n = X.shape[0]
        labels = np.full(n, -1)
        visited = np.zeros(n, dtype=bool)
        cluster_id = 0

        nn = NearestNeighbors(radius=self.eps)
        nn.fit(X)
        neighborhoods = nn.radius_neighbors(X, return_distance=False)

        for i in range(n):
            if visited[i]:
                continue

            visited[i] = True
            neighbors = neighborhoods[i]

            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            labels[i] = cluster_id
            seeds = list(neighbors)

            j = 0
            while j < len(seeds):
                point = seeds[j]

                if not visited[point]:
                    visited[point] = True
                    point_neighbors = neighborhoods[point]

                    if len(point_neighbors) >= self.min_samples:
                        seeds.extend(point_neighbors)

                if labels[point] == -1:
                    labels[point] = cluster_id

                j += 1

            cluster_id += 1

        self.labels_ = labels
        return labels

    def plot_clusters(self, X):
        if self.labels_ is None:
            raise ValueError("DBSCAN must be fitted before plotting.")

        # Count clusters (exclude noise = -1)
        unique_clusters = [c for c in np.unique(self.labels_) if c != -1]
        n_clusters = len(unique_clusters)

        # Dynamic filename
        filename = f"dbscan_eps{self.eps}_min{self.min_samples}_clusters{n_clusters}.png"
        save_path = os.path.join(self.save_dir, filename)

        # Plot
        plt.figure(figsize=(8, 6))
        plt.scatter(
            X[:, 0], X[:, 1],
            c=self.labels_,
            cmap="tab10",
            s=20
        )
        plt.title(f"DBSCAN (eps={self.eps}, min_samples={self.min_samples})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"[INFO] Plot saved to: {save_path}")
