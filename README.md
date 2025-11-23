# Wine Quality Dataset - PCA & Clustering Analysis

## Project Description
This project performs an exploratory analysis of the Wine Quality dataset using Principal Component Analysis (PCA) and K-means clustering. The goal is to identify patterns in the chemical composition of wines, separate them into clusters, and determine which chemical features statistically differentiate these clusters using ANOVA. Additionally, wine IDs are retrieved per cluster for traceability.

---

## Packages Used

| Package | Purpose |
|---------|---------|
| `pandas` | Data manipulation and handling tabular datasets |
| `numpy` | Numerical operations and array handling |
| `matplotlib` | Data visualization (scatter plots, line plots) |
| `seaborn` | Enhanced visualization (heatmaps, boxplots) |
| `scikit-learn` (`StandardScaler`, `PCA`, `KMeans`, `silhouette_score`) | Standardization, dimensionality reduction, clustering, cluster evaluation |
| `scipy.stats` (`f_oneway`) | Statistical analysis, ANOVA testing |

---

## Code Workflow and Results

### 1. Load Dataset
- Dataset: `WineQT.csv` containing chemical properties, wine quality scores, and unique IDs.  

---

### 2. Preprocessing
- Dropped `Id` column for analysis.  
- Removed missing values.  
- Plotted correlation matrix to explore relationships between chemical variables.

---

### 3. Features and Standardization
- Separated features (`X`) and target (`y`).  
- Standardized features using `StandardScaler` to normalize scales.

---

### 4. PCA
- Applied PCA with 2 components for visualization.  
- Visualized PCA scatter plot colored by wine quality.

**Result:** 
- The first two components together explain only ~46% of the variance, so the 2D plot provides a limited but useful view of the data.
- No clear clusters or trends are visible in the PCA scatter plot, indicating that chemical properties are distributed relatively uniformly and that clustering is not trivially visible in the first two components.

---

### 5. K-means Clustering
- Determined optimal number of clusters `k` using silhouette scores.  
- Applied K-means clustering on PCA components.  
- Visualized clusters in PCA space with centroids.

**Result:**
- Wines are grouped into 2 distinct clusters, capturing meaningful differences in chemical composition despite the PCA scatter plot not showing obvious aggregations.

---

### 6. Cluster Interpretation
- Added cluster labels to dataset and summarized mean values per cluster.

**Result:** 
- Clusters differ significantly in variables such as `fixed acidity`, `citric acid`, `pH`, and `density`.

---

### 7. Retrieve Wine IDs per Cluster
- Mapped cluster labels to original dataset to recover wine IDs per cluster.  
- Saved updated dataset with cluster assignments (`wine_with_clusters.csv`).

**Result:**
- Eeasy identification of which wines belong to each cluster.

---

### 8. ANOVA Analysis
- Performed one-way ANOVA for each chemical variable to test if means differ significantly between clusters.  
- Highlighted variables with p-value < 0.05 as statistically significant.

**Result:**
- The ANOVA confirms that all chemical properties contribute to the separation of the two clusters.
- This validates the cluster interpretation and highlights which features drive the differentiation between wine groups.

---

## Key Findings
- PCA provides a clear 2D visualization of wine distribution by quality.  
- K-means clusters wines based on chemical composition; clusters align with significant chemical differences.  
- ANOVA confirms which variables are statistically responsible for separating the clusters.  
- Wine IDs can be tracked per cluster, enabling further analysis or targeted experiments.

---

## Author

Alessandro Bifulco
