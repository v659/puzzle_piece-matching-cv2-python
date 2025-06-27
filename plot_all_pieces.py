import ast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN

# File with Harris points
file_path = "file.txt"  # Update as needed

# Parameters
dedup_radius = 5  # Cluster radius in pixels

# Parse Harris points
harris_points_all = []
with open(file_path, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    if line.startswith("==="):
        continue
    if line.startswith("[") and line.endswith("]"):
        try:
            points = ast.literal_eval(line)
            if isinstance(points, list) and all(isinstance(p, tuple) and len(p) == 2 for p in points):
                harris_points_all.append(points)
        except Exception as e:
            print(f"Failed parsing: {line[:80]}... -> {e}")

# Deduplicate using DBSCAN clustering
def deduplicate_points(points, radius=5):
    if not points:
        return []
    X = np.array(points)
    clustering = DBSCAN(eps=radius, min_samples=1).fit(X)
    labels = clustering.labels_

    deduped = []
    for label in np.unique(labels):
        cluster_points = X[labels == label]
        centroid = cluster_points.mean(axis=0).astype(int)
        deduped.append((centroid[0], centroid[1]))
    return deduped

# Plot deduplicated points
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, (original_points, ax) in enumerate(zip(harris_points_all, axes)):
    filtered_points = deduplicate_points(original_points, dedup_radius)
    x_vals, y_vals = zip(*filtered_points)
    ax.scatter(x_vals, y_vals, s=3, color='green', alpha=0.7)
    ax.set_title(f"Piece {idx}: {len(original_points)} raw → {len(filtered_points)} deduped")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.invert_yaxis()
    ax.grid(True)

# Hide unused plots
for j in range(len(harris_points_all), len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()
