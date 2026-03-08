# Customer Segmentation using K-Means Clustering

## Task

Create a K-Means clustering algorithm to group customers of a retail store based on their purchase history.

## Dataset

- **Mall_Customers.csv** — 200 customer records with features: CustomerID, Gender, Age, Annual Income (k$), Spending Score (1-100).

## Pipeline

### 1. K-Means Clustering (`kmeans_clustering.py`)

- Selects `Annual Income` and `Spending Score` as clustering features.
- Uses the **Elbow Method** to determine optimal clusters (K=5).
- Visualizes customer segments and saves plots.

### 2. ML Classification Pipeline (`ml_pipeline.py`)

- **Feature Engineering**: Creates `Income_to_Age_Ratio`, `Spending_to_Income_Ratio`, encodes `Gender`.
- **Data Splitting**: 70% Train / 15% Validation / 15% Test with `StandardScaler`.
- **Baseline Model**: Logistic Regression with L2 regularization.
- **Hyperparameter Tuning**: Grid Search over regularization strength `C`.
- **Evaluation**: 5-Fold Cross-Validation (**96.47%**), Test Set Accuracy (**96.67%**).

## Results

| Metric | Value |
|---|---|
| Optimal Clusters | 5 |
| Cross-Validation Accuracy | 96.47% |
| Test Set Accuracy | 96.67% |

## Outputs

- `elbow_method.png` — Elbow curve plot
- `clusters_original.png` — Customer clusters (original scale)
- `clusters_scaled.png` — Customer clusters (scaled)
- `confusion_matrix.png` — Classification confusion matrix
- `Mall_Customers_Clustered.csv` — Dataset with cluster labels
