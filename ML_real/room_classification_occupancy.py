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


feature_columns = ["co2","#occupants"]

assembler = ColumnTransformer(
    transformers=[
        ("features", FunctionTransformer(lambda x: x, validate=False), 
         feature_columns)
    ]
)
assembled_df=merged_df
# Transform the DataFrame
assembled_df["features"] = list(assembler.fit_transform(merged_df))

# Standardize the features
features_matrix = np.vstack(merged_df["features"])

# Initialize and apply StandardScaler
scaler_model = StandardScaler()
scaled_features = scaler_model.fit_transform(features_matrix)

scaled_df = assembled_df
scaled_df["scaledFeatures"] = list(scaled_features)

sil_scores = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    sil_scores.append(score)

plt.plot(K, sil_scores, 'go-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for different k')
plt.grid(True)
plt.savefig("k-values.png")

# Apply KMeans
# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_model = kmeans.fit(scaled_features)
clustered_df = scaled_df
clustered_df["prediction"] = kmeans_model.predict(scaled_features)
score = silhouette_score(scaled_features, kmeans_model.labels_)
print(f"Silhouette Score: {score}")
# Select specific columns
final_df = clustered_df[[ "co2", "#occupants", "prediction"]]


# Extract cluster ranges
cluster_ranges = final_df.groupby("prediction").agg(
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
    cluster_plot_df,
    vars=["co2","#occupants"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("Cluster Visualization by Occupancy", y=1.02)

# Save the plot
pairplot.savefig("cluster_pairplot_occupants.png")
plt.show()
