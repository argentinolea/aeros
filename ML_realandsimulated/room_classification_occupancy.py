import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

def train_regression_models_by_cluster(clustered_df):
    cluster_models = {}
    cluster_metrics = {}
    negative_r2_clusters = []

    feature_cols = ["temperature", "humidity", "ventilation rate", "volume"]
    target_col = "co2"

    for cluster_id in clustered_df["prediction"].unique():
        print(f"\n🔬 Training Linear Regression for Cluster {cluster_id}")
        cluster_data = clustered_df[clustered_df["prediction"] == cluster_id].copy()

        X = cluster_data[feature_cols]
        y = cluster_data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        if r2 < 0:
            negative_r2_clusters.append(cluster_id)

        print(f"   MAE  (Mean Absolute Error)      : {mae:.2f} ppm")
        print(f"   MSE  (Mean Squared Error)       : {mse:.2f} ppm²")
        print(f"   RMSE (Root Mean Squared Error)  : {rmse:.2f} ppm")
        print(f"   R²   (Coefficient of Determination): {r2:.4f}")

        cluster_models[cluster_id] = model
        cluster_metrics[cluster_id] = {
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2
        }

    return cluster_models, cluster_metrics, negative_r2_clusters

def validate_sensor_against_all_clusters(sensor_data, cluster_ranges, cluster_models):
    # Extract feature values from the single sensor data row
    sensor = sensor_data.iloc[0]  # safely extract scalar values
    t = sensor["temperature"]
    h = sensor["humidity"]
    v = sensor["ventilation rate"]
    vol = sensor["volume"]
    co2 = sensor["co2"]

    matched_clusters = []
    feature_vector = np.array([[t, h, v, vol]])
    for cluster_id, row in cluster_ranges.iterrows():

        if (
            row["min_temperature"] <= t <= row["max_temperature"] and
            row["min_humidity"] <= h <= row["max_humidity"] and
            row["min_ventilation"] <= v <= row["max_ventilation"] and
            row["min_volume"] <= vol <= row["max_volume"]
        ):
            model = cluster_models.get(cluster_id)
            if model:
                prediction = model.predict(feature_vector)[0]
                error = abs(co2 - prediction)

                matched_clusters.append({
                    "co2": co2,
                    "lr_prediction": prediction,
                    "error": error,
                    "cluster_id": cluster_id
                })

    if not matched_clusters:
        print("❌ No matching cluster found for sensor data.")
        return

    print("✅ Sensor matched the following clusters:")
    for match in matched_clusters:
        if match["lr_prediction"] < 0:
            continue

        occ_range = cluster_ranges.loc[match["cluster_id"]]
        min_occ = occ_range["min_occupants"]
        max_occ = occ_range["max_occupants"]

        print(f"  Cluster {match['cluster_id']}:")
        print(f"     CO₂ actual   : {match['co2']:.2f}")
        print(f"     CO₂ predicted: {match['lr_prediction']:.2f}")
        print(f"     Error        : {match['error']:.2f}")
        print(f"     min_occupants_modified: {min_occ}")
        print(f"     max_occupants_modified: {max_occ}")

##############################
# 1. CARICAMENTO DEI DATI
##############################

# Carica dati reali e simulati
input_file_path_real = "../co2_false_negative_real/10022002_modified.csv"
data_df_real = pd.read_csv(input_file_path_real, delimiter=",")
merged_df_real = data_df_real[["temperature", "co2", "humidity", "volume", "ventilation rate", "occupants_modified"]].round(2)
input_file_path_sim = "../co2_false_negative_sim/65m3_export_false_negative.csv"
data_df_sim = pd.read_csv(input_file_path_sim, delimiter=",")
merged_df_sim = data_df_sim[["temperature", "co2", "humidity", "volume", "ventilation rate", "occupants_modified"]].round(2)
merged_df_sim["occupants_modified"] = merged_df_sim["occupants_modified"].astype(bool)
print(merged_df_sim)
merged_df = pd.concat([merged_df_real, merged_df_sim], ignore_index=True)

##############################
# 2. PRETRAINING (SU DATI SIMULATI)
##############################

# Calcolo della varianza (filtraggio per varianza CO2 tra 0 e 20)

merged_df = merged_df[[
    "temperature", "co2", "humidity",
    "volume", "ventilation rate", "occupants_modified"
]].round(2)

merged_df = merged_df[
    (merged_df["temperature"] > 20) & (merged_df["temperature"] < 25) &
    (merged_df["humidity"] > 20) & (merged_df["humidity"] < 80) &
    (merged_df["volume"] > 55) & (merged_df["volume"] < 75)
]

variance_df = merged_df.groupby([
    "occupants_modified"
]).agg(
    CO2_variance=("co2", "var")
).dropna().reset_index()


feature_columns = ["CO2_variance","occupants_modified"]

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
final_df = clustered_df[["CO2_variance", "occupants_modified", "prediction"]]

merged_with_clusters = pd.merge(
    merged_df,                     # original data with co2
    final_df[["occupants_modified", "prediction"]],  # clustering result
    on="occupants_modified",
    how="inner"
)

# Extract cluster ranges
cluster_ranges = merged_with_clusters.groupby("prediction").agg(
    min_co2=("co2", "min"),
    max_co2=("co2", "max"),
    min_occupants=("occupants_modified", "min"),
    max_occupants=("occupants_modified", "max")
)

print("cluster_ranges")
print(cluster_ranges)

# Convert Spark DataFrame to Pandas
cluster_plot_df = final_df

sns.set(style="whitegrid")
pairplot = sns.pairplot(
    merged_with_clusters,
    vars=["co2","occupants_modified"],
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
    x="occupants_modified",
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
