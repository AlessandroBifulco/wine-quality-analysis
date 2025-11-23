## Wine Quality Dataset - PCA & Clustering Analysis

# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway

# 2. Load dataset

df = pd.read_csv("WineQT.csv")
df_original = df.copy()

print("Dataset shape:", df.shape)
print(df.head())

# 3. Basic cleaning & preprocessing

df = df.drop(columns=["Id"])
print("\nMissing values:\n", df.isnull().sum())
df = df.dropna()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Matrix")
plt.show()

# 4. Separate features and target

X = df.drop(columns=["quality"])   
y = df["quality"]                  

# 5. Data Standardization

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. PCA

pca = PCA(n_components=2)   
X_pca = pca.fit_transform(X_scaled)
print("\nExplained variance ratio per component:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap="viridis", alpha=0.7)
plt.colorbar(label="Quality")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Wine Data (colored by quality)")
plt.show()

# 7. K-means clustering

sil_scores = []
K_range = range(2, 8)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)

plt.figure(figsize=(8,5))
plt.plot(K_range, sil_scores, marker="o")
plt.title("Silhouette Score vs Number of Clusters")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()

best_k = K_range[np.argmax(sil_scores)]
print("Best number of clusters:", best_k)

kmeans = KMeans(n_clusters=best_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# 8. Visualize clusters in PCA space

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=clusters, cmap="tab10", alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
            s=200, c="black", marker="X", label="Centroids")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"K-means Clustering (k={best_k}) on PCA Components")
plt.legend()
plt.show()

# 9. Cluster interpretation

df_clusters = df.copy()
df_clusters["cluster"] = clusters

cluster_summary = df_clusters.groupby("cluster").mean()
print("\nCluster summary statistics:")
print(cluster_summary)

# 10. Retrieve wine IDs for each cluster

df_original["cluster"] = clusters
cluster0_ids = df_original.loc[df_original["cluster"] == 0, "Id"].tolist()
cluster1_ids = df_original.loc[df_original["cluster"] == 1, "Id"].tolist()

print(f"\nNumber of wines in Cluster 0: {len(cluster0_ids)}")
print(f"IDs of first 10 wines in Cluster 0: {cluster0_ids[:10]}")
print(f"\nNumber of wines in Cluster 1: {len(cluster1_ids)}")
print(f"IDs of first 10 wines in Cluster 1: {cluster1_ids[:10]}")

df_original.to_csv("wine_with_clusters.csv", index=False)

# 11. ANOVA: Test which variables differ significantly between clusters

features = X.columns
anova_results = {}

for feature in features:
    cluster0_vals = df_original.loc[df_original["cluster"] == 0, feature]
    cluster1_vals = df_original.loc[df_original["cluster"] == 1, feature]
    
    f_stat, p_val = f_oneway(cluster0_vals, cluster1_vals)
    anova_results[feature] = {"F-statistic": f_stat, "p-value": p_val}

anova_df = pd.DataFrame(anova_results).T.sort_values("p-value")
print("\nANOVA results (sorted by p-value):")
print(anova_df)

significant_vars = anova_df[anova_df["p-value"] < 0.05].index.tolist()
print("\nVariables significantly different between clusters (p < 0.05):")
print(significant_vars)