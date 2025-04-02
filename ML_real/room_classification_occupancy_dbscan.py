import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score

def plot_k_distance(data, k=4):
    """
    Plots the distance to the k-th nearest neighbor for each point.
    Helps choose the `eps` parameter for DBSCAN.
    
    Parameters:
    - data: scaled feature array (e.g., output of StandardScaler)
    - k: typically equal to min_samples in DBSCAN
    """
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(data)
    
    distances, indices = nbrs.kneighbors(data)
    # Get the distance to the kth neighbor
    k_distances = np.sort(distances[:, k-1])
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_distances)
    plt.ylabel(f"{k}th Nearest Neighbor Distance")
    plt.xlabel("Points sorted by distance")
    plt.title(f"K-distance plot (k = {k})")
    plt.grid(True)
    plt.savefig("k-distance-plot.png")
    plt.show()
    

# Load data from CSV
input_file_path = "../capture/10022002_fixed_all_features.csv"
data_df = pd.read_csv(input_file_path, delimiter=";")

# Feature selection and rounding
merged_df = data_df[["co2", "#occupants"]].round(2)
feature_columns = ["co2", "#occupants"]

# Assemble features
assembler = ColumnTransformer(
    transformers=[
        ("features", FunctionTransformer(lambda x: x, validate=False), feature_columns)
    ]
)
assembled_df = merged_df.copy()
assembled_df["features"] = list(assembler.fit_transform(merged_df))

# Convert to NumPy array
features_matrix = np.vstack(assembled_df["features"])

# Standardize features
scaler_model = StandardScaler()
scaled_features = scaler_model.fit_transform(features_matrix)
plot_k_distance(scaled_features, k=5)
scaled_df = assembled_df
scaled_df["scaledFeatures"] = list(scaled_features)

print("DBSCAN algorithm")
# Apply DBSCAN
dbscan = DBSCAN(eps=0.05, min_samples=10)
if np.isnan(scaled_features).any() or np.isinf(scaled_features).any():
    print("❌ Error: scaled_features contains NaN or infinite values.")
else:
    print("DBSCAN algorithm:fit")
    dbscan_model = dbscan.fit(scaled_features)
dbscan_model = dbscan.fit(scaled_features)

print("DBSCAN algorithm:scaled_df.copy")
# Add predictions
clustered_df = scaled_df
clustered_df["prediction"] = dbscan_model.labels_

# Silhouette Score (only if more than 1 cluster, ignoring noise)
mask = dbscan_model.labels_ != -1
if len(set(dbscan_model.labels_)) > 1 and mask.sum() > 0:
    score = silhouette_score(scaled_features[mask], dbscan_model.labels_[mask])
    print(f"Silhouette Score (DBSCAN): {score}")
else:
    print("Silhouette Score (DBSCAN): Not applicable (too few clusters or only noise)")

# Count noise points
n_noise = np.sum(dbscan_model.labels_ == -1)
print(f"Number of noise points: {n_noise}")

# Select specific columns
final_df = clustered_df[["co2", "#occupants", "prediction"]]

# Extract cluster ranges (exclude noise if needed)
cluster_ranges = final_df[final_df["prediction"] != -1].groupby("prediction").agg(
    min_co2=("co2", "min"),
    max_co2=("co2", "max"),
    min_occupants=("#occupants", "min"),
    max_occupants=("#occupants", "max")
)
print("cluster_ranges")
print(cluster_ranges)

# Plot
sns.set(style="whitegrid")
pairplot = sns.pairplot(
    final_df,
    vars=["co2", "#occupants"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("DBSCAN Cluster Visualization by Occupancy", y=1.02)

# Save the plot
pairplot.savefig("dbscan_cluster_pairplot_occupants_dbscan.png")
plt.show()