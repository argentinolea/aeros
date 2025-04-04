import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
        
# Load data from CSV
input_file_path = "../capture/10022002_fixed_all_features.csv"
data_df = pd.read_csv(input_file_path, delimiter=";")
# Feature selection and rounding
merged_df = data_df[[
    "co2",  "#occupants"
]].round(2)

variance_df = merged_df.groupby([
    "#occupants"
]).agg(
    CO2_variance=("co2", "var")
).dropna().reset_index()


feature_columns = ["CO2_variance","#occupants"]

assembler = ColumnTransformer(
    transformers=[
        ("features", FunctionTransformer(lambda x: x, validate=False), 
         feature_columns)
    ]
)
assembled_df=variance_df
# Transform the DataFrame
assembled_df["features"] = list(assembler.fit_transform(variance_df))

# Standardize the features
features_matrix = np.vstack(variance_df["features"])

# Initialize and apply StandardScaler
scaler_model = StandardScaler()
scaled_features = scaler_model.fit_transform(features_matrix)
print(scaled_features)

scaled_df = assembled_df
scaled_df["scaledFeatures"] = list(scaled_features)

sil_scores = []
valid_k = []
K = range(2, 11)
n_samples = scaled_features.shape[0]

for k in K:
    if k >= n_samples:
        print(f"Skipping k={k} because it's >= n_samples ({n_samples})")
        continue
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    sil_scores.append(score)
    valid_k.append(k)

plt.plot(valid_k, sil_scores, 'go-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for different k')
plt.grid(True)
plt.savefig("k-values.png")

# Apply KMeans
# Apply KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_model = kmeans.fit(scaled_features)
clustered_df = scaled_df
clustered_df["prediction"] = kmeans_model.predict(scaled_features)
score = silhouette_score(scaled_features, kmeans_model.labels_)
print(f"Silhouette Score: {score}")
# Select specific columns
final_df = clustered_df[["CO2_variance", "#occupants", "prediction"]]

merged_with_clusters = pd.merge(
    merged_df,                     # original data with co2
    final_df[["#occupants", "prediction"]],  # clustering result
    on="#occupants",
    how="inner"
)

# Extract cluster ranges
cluster_ranges = merged_with_clusters.groupby("prediction").agg(
    min_co2=("co2", "min"),
    max_co2=("co2", "max"),
    min_occupants=("#occupants", "min"),
    max_occupants=("#occupants", "max")
)

print("cluster_ranges")
print(cluster_ranges)

# Convert Spark DataFrame to Pandas
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
pairplot.savefig("cluster_pairplot_occupants.png")
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
plt.savefig("barplot_co2_by_occupancy_and_cluster.png")
plt.show()
