import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# Load data
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
scaled_df = assembled_df.copy()
scaled_df["scaledFeatures"] = list(scaled_features)

# Apply GMM
print("GMM algorithm")
gmm = GaussianMixture(n_components=4, random_state=42)
gmm.fit(scaled_features)

# Predict cluster labels
gmm_labels = gmm.predict(scaled_features)

# Add predictions
clustered_df = scaled_df.copy()
clustered_df["prediction"] = gmm_labels

# Silhouette Score
if len(set(gmm_labels)) > 1:
    score = silhouette_score(scaled_features, gmm_labels)
    print(f"Silhouette Score (GMM): {score}")
else:
    print("Silhouette Score (GMM): Not applicable (only one cluster)")

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
pairplot.savefig("gmm_cluster_pairplot_occupants.png")
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
plt.savefig("gmm_barplot_co2_by_occupancy_and_cluster.png")
plt.show()