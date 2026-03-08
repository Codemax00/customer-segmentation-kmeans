import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def draw_cluster_ellipse(ax, points, color, n_std=2.0):
    """Draw a confidence ellipse around a cluster of 2D points."""
    if len(points) < 2:
        return
    mean = np.mean(points, axis=0)
    cov = np.cov(points, rowvar=False)
    # Eigenvalues & eigenvectors of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by largest eigenvalue
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    # Angle of rotation
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    # Width and height of ellipse (scaled by n_std standard deviations)
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle,
                      edgecolor=color, facecolor=color, alpha=0.15, linewidth=2)
    ax.add_patch(ellipse)

# Load data
df = pd.read_csv('Mall_Customers.csv')

# Select relevant features based on purchase history
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# It's good practice to scale features for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding the optimal number of clusters using Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Save Elbow Method plot
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.savefig('elbow_method.png')
plt.close()

# Looking at the elbow plot, 5 is the optimal number of clusters for this dataset
optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original dataframe
df['Cluster'] = y_kmeans

# Visualize the clusters with ellipses
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(optimal_clusters):
    cluster_points = X_scaled[y_kmeans == i]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    draw_cluster_ellipse(ax, cluster_points, colors[i])

ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='yellow', label='Centroids', marker='*', edgecolors='black', zorder=5)
ax.set_title('Clusters of Customers')
ax.set_xlabel('Annual Income (k$) (Scaled)')
ax.set_ylabel('Spending Score (1-100) (Scaled)')
ax.legend()
plt.savefig('clusters_scaled.png')
plt.close()

# Visualize the clusters on original unscaled data with ellipses
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(optimal_clusters):
    cluster_points = X.iloc[y_kmeans == i].values
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], s=50, c=colors[i], label=f'Cluster {i+1}')
    draw_cluster_ellipse(ax, cluster_points, colors[i])

# Inverse transform centroids to original scale
centroids_original = scaler.inverse_transform(kmeans.cluster_centers_)
ax.scatter(centroids_original[:, 0], centroids_original[:, 1], s=200, c='yellow', label='Centroids', marker='*', edgecolors='black', zorder=5)
ax.set_title('Clusters of Customers (Original Scale)')
ax.set_xlabel('Annual Income (k$)')
ax.set_ylabel('Spending Score (1-100)')
ax.legend()
plt.savefig('clusters_original.png')
plt.close()

# Print summary
print('K-Means clustering completed successfully.')
print(f'Optimal number of clusters chosen: {optimal_clusters}')
print('\nCluster summary (average values):')
cluster_summary = df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean().round(2)
print(cluster_summary)

# Save final clustered data
df.to_csv('Mall_Customers_Clustered.csv', index=False)
print('\nClustered data saved to Mall_Customers_Clustered.csv')
