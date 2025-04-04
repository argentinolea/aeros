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
variance_df = merged_df.groupby([
    "#occupants"
]).agg(
    CO2_variance=("co2", "var")
).dropna().reset_index()
feature_columns = ["CO2_variance", "#occupants"]

# Assemble features
assembler = ColumnTransformer(
    transformers=[
        ("features", FunctionTransformer(lambda x: x, validate=False), feature_columns)
    ]
)
assembled_df=variance_df
assembled_df["features"] = list(assembler.fit_transform(variance_df))

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
dbscan = DBSCAN(eps=50, min_samples=2)
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
labels = dbscan_model.labels_
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise} out of {len(labels)} total points")

# Select specific columns
final_df = clustered_df[["CO2_variance", "#occupants", "prediction"]]

merged_with_clusters = pd.merge(
    merged_df,                     # original data with co2
    final_df[["#occupants", "prediction"]],  # clustering result
    on="#occupants",
    how="inner"
)

# Extract cluster ranges (exclude noise if needed)
cluster_ranges = merged_with_clusters.groupby("prediction").agg(
    min_co2=("co2", "min"),
    max_co2=("co2", "max"),
    min_occupants=("#occupants", "min"),
    max_occupants=("#occupants", "max")
)
print("cluster_ranges")
print(cluster_ranges)

cluster_plot_df = final_df

sns.set(style="whitegrid")
pairplot = sns.pairplot(
    merged_with_clusters,
    vars=["co2","#occupants"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("Cluster Visualization by Occupancy", y=1.02)

# Save the plot
pairplot.savefig("dbscan_cluster_pairplot_occupants.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(
    data=merged_with_clusters,
    x="#occupants",
    y="co2",
    hue="prediction",
    palette="tab10",
    errorbar="sd"  # show standard deviation as error bar
)
plt.title("Average CO₂ by Occupancy and Cluster")
plt.xlabel("Occupants")
plt.ylabel("Average CO₂ (ppm)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("dbscan_barplot_co2_by_occupancy_and_cluster.png")
plt.show()