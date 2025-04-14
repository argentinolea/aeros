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
import warnings
warnings.filterwarnings("ignore")

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
        
        plt.figure(figsize=(8, 6))

        # Scatter: True vs Predicted
        plt.scatter(y_test, y_pred, color='green', alpha=0.5, s=30, label="Actual CO₂")

        # Identity line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], color='blue', linewidth=2, alpha=0.5, label="Ideal Fit")

        plt.xlabel("True CO₂ Concentration")
        plt.ylabel("Predicted CO₂ Concentration")
        plt.title(f"Linear Regression Fit for Cluster {cluster_id}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"lr_fit_cluster_{cluster_id}.png")
        plt.show()

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

# Aggiungiamo la feature CO2_variance e definiamo le feature da usare
feature_columns = ["temperature", "humidity", "ventilation rate", "volume", "occupants_modified"]

assembler = ColumnTransformer(
    transformers=[
        ("features", FunctionTransformer(lambda x: x, validate=False), feature_columns)
    ]
)

assembled_df=merged_df

assembled_df["features"] = list(assembler.fit_transform(merged_df))

# Standardize the features
features_matrix = np.vstack(merged_df["features"])

# Initialize and apply StandardScaler
scaler_model = StandardScaler()
scaled_features = scaler_model.fit_transform(features_matrix)

scaled_df = assembled_df
scaled_df["scaledFeatures"] = list(scaled_features)

# Applichiamo la trasformazione
assembled_df = merged_df.copy()
assembled_df["features"] = list(assembler.fit_transform(merged_df))

# Standardizzazione
features_matrix = np.vstack(assembled_df["features"])
scaler_model = StandardScaler()
scaled_features = scaler_model.fit_transform(features_matrix)
assembled_df["scaledFeatures"] = list(scaled_features)

# Applica KMeans (usando solo dati simulati)
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_model = kmeans.fit(scaled_features)
clustered_df = scaled_df
clustered_df["prediction"] = kmeans_model.predict(scaled_features)

score = silhouette_score(scaled_features, kmeans_model.labels_)
print(f"Silhouette Score: {score}")
cluster_models, cluster_metrics, negative_r2_clusters = train_regression_models_by_cluster(clustered_df)
clustered_df = clustered_df[~clustered_df["prediction"].isin(negative_r2_clusters)]

# Select specific columns
final_df = clustered_df[["temperature", "humidity", "ventilation rate", "volume", "occupants_modified", "prediction"]]

# Estraiamo gli intervalli per ciascun cluster (cluster ranges) dai dati simulati
cluster_ranges = final_df.groupby("prediction").agg(
    min_temperature=("temperature", "min"),
    max_temperature=("temperature", "max"),
    min_volume=("volume", "min"),
    max_volume=("volume", "max"),
    min_ventilation=("ventilation rate", "min"),
    max_ventilation=("ventilation rate", "max"),
    min_humidity=("humidity", "min"),
    max_humidity=("humidity", "max"),
    min_occupants=("occupants_modified", "min"),
    max_occupants=("occupants_modified", "max")
).reset_index()

print("cluster_ranges:")
print(cluster_ranges)

# Convert Spark DataFrame to Pandas
cluster_plot_df = final_df

sns.set(style="whitegrid")
pairplot = sns.pairplot(
    cluster_plot_df,
    vars=["temperature", "humidity", "occupants_modified"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("Cluster Visualization by Environmental Features and Occupancy", y=1.02)

# Save the plot
pairplot.savefig("cluster_pairplot_temperature_humidity_occupants.png")
plt.show()


sensor_assembler = ColumnTransformer(
    transformers=[
        ("features", FunctionTransformer(lambda x: x, validate=False), 
        ["temperature", "humidity", "ventilation rate", "volume"]) 
    ]
)
    
sensor_data_1 = pd.DataFrame([
    {"temperature": 22.35, "humidity": 43.5, "ventilation rate": 0.25, "volume": 65.0, "co2": 1150.0}
])


validate_sensor_against_all_clusters(
    sensor_data=sensor_data_1,
    cluster_ranges=cluster_ranges,
    cluster_models=cluster_models
)

sensor_data_2 = pd.DataFrame([
    {"temperature": 22.35, "humidity": 43.5, "ventilation rate": 0.25, "volume": 65.0, "co2": 5000.0}
])


validate_sensor_against_all_clusters(
    sensor_data=sensor_data_2,
    cluster_ranges=cluster_ranges,
    cluster_models=cluster_models
)
