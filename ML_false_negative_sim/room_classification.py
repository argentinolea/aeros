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

def train_regression_models_by_cluster(clustered_df):
    cluster_models = {}
    cluster_metrics = {}
    negative_r2_clusters = []

    feature_cols = ["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"]
    target_col = "co2"

    for cluster_id in clustered_df["prediction"].unique():
        cluster_data = clustered_df[clustered_df["prediction"] == cluster_id].copy()

        X = cluster_data[feature_cols]
        y = cluster_data[target_col]

        if len(cluster_data) < 2:
            print(f"⚠️ Skipping Cluster {cluster_id}: Not enough samples for training.")
            continue
        print(f"\n🔬 Training Linear Regression for Cluster {cluster_id}")
        
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
    t = sensor["Zone Mean Air Temperature"]
    h = sensor["Zone Air Relative Humidity"]
    v = sensor["Ventilation"]
    vol = sensor["_volume"]
    co2 = sensor["co2"]

    matched_clusters = []
    feature_vector = np.array([[t, h, v, vol]])
    print (cluster_ranges)
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
    for match in matched_clusters :
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
        
# Load data from CSV
input_file_path = "../co2_false_negative_sim/65m3_export_false_negative.csv"
data_df = pd.read_csv(input_file_path, delimiter=",")
# Feature selection and rounding
merged_df = data_df[[
    "Zone Mean Air Temperature", "co2", "Zone Air Relative Humidity",
    "_volume", "Ventilation", "occupants_modified"
]].round(2)


merged_df = merged_df[
                    (merged_df["Zone Mean Air Temperature"] > 20) &
                    (merged_df["Zone Mean Air Temperature"] < 40) &
                    (merged_df["Zone Air Relative Humidity"] > 20) &
                    (merged_df["Zone Air Relative Humidity"] < 80) &
                    (merged_df["_volume"] > 55) &
                    (merged_df["_volume"] < 75)]

feature_columns = ["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "occupants_modified"]

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

# Apply KMeans
# Apply KMeans clustering
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans_model = kmeans.fit(scaled_features)
clustered_df = scaled_df
clustered_df["prediction"] = kmeans_model.predict(scaled_features)
cluster_models, cluster_metrics, negative_r2_clusters = train_regression_models_by_cluster(clustered_df)
clustered_df = clustered_df[~clustered_df["prediction"].isin(negative_r2_clusters)]

# Select specific columns
final_df = clustered_df[["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "occupants_modified", "prediction"]]


# Extract cluster ranges
cluster_ranges = final_df.groupby("prediction").agg(
    min_temperature=("Zone Mean Air Temperature", "min"),
    max_temperature=("Zone Mean Air Temperature", "max"),
    min_volume=("_volume", "min"),
    max_volume=("_volume", "max"),
    min_ventilation=("Ventilation", "min"),
    max_ventilation=("Ventilation", "max"),
    min_humidity=("Zone Air Relative Humidity", "min"),
    max_humidity=("Zone Air Relative Humidity", "max"),
    min_occupants=("occupants_modified", "min"),
    max_occupants=("occupants_modified", "max")
)

print("cluster_ranges")
print(cluster_ranges)

# Convert Spark DataFrame to Pandas
cluster_plot_df = final_df

cluster_plot_df = cluster_plot_df[cluster_plot_df.groupby("prediction")["prediction"].transform("count") >= 2]

sns.set(style="whitegrid")
pairplot = sns.pairplot(
    cluster_plot_df,
    vars=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "occupants_modified"],
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
        ["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"]) 
    ]
)
    
sensor_data_1 = pd.DataFrame([
    {"Zone Mean Air Temperature": 20.15, "Zone Air Relative Humidity": 26.5, "Ventilation": 0.00, "_volume": 68.56, "co2": 1150.0}
])


validate_sensor_against_all_clusters(
    sensor_data=sensor_data_1,
    cluster_ranges=cluster_ranges,
    cluster_models=cluster_models
)

sensor_data_2 = pd.DataFrame([
    {"Zone Mean Air Temperature": 20.15, "Zone Air Relative Humidity": 26.5, "Ventilation": 0.00, "_volume": 68.56, "co2": 5000.0}
])


validate_sensor_against_all_clusters(
    sensor_data=sensor_data_2,
    cluster_ranges=cluster_ranges,
    cluster_models=cluster_models
)
