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
final_df = clustered_df[["co2", "#occupants", "prediction"]]

# Extract cluster ranges
cluster_ranges = final_df.groupby("prediction").agg(
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
pairplot.fig.suptitle("GMM Cluster Visualization by Occupancy", y=1.02)

# Save the plot
pairplot.savefig("gmm_cluster_pairplot_occupants_gmm.png")
plt.show()