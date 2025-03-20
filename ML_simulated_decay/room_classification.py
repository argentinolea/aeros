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

def assign_to_cluster(kmeans_model, scaler_model, clustering_assembler, sensor_data):
    sensor_data["CO2_variance"] = 0.0  # Add a default CO2 variance column
    assembled_sensor_data = clustering_assembler.transform(sensor_data)
    scaled_sensor_data = scaler_model.transform(assembled_sensor_data)
    cluster = kmeans_model.predict(scaled_sensor_data)
    return cluster[0], scaled_sensor_data

def train_regression_for_cluster(cluster_id, cluster_ranges, merged_df):
    cluster_range = cluster_ranges[cluster_ranges["prediction"] == cluster_id].iloc[0]
    
    cluster_data = merged_df[(merged_df["Zone Mean Air Temperature"] >= cluster_range["min_temperature"]) &
                             (merged_df["Zone Mean Air Temperature"] <= cluster_range["max_temperature"]) &
                             (merged_df["_volume"] >= cluster_range["min_volume"]) &
                             (merged_df["_volume"] <= cluster_range["max_volume"]) &
                             (merged_df["Ventilation"] >= cluster_range["min_ventilation"]) &
                             (merged_df["Ventilation"] <= cluster_range["max_ventilation"]) &
                             (merged_df["Zone Air Relative Humidity"] >= cluster_range["min_humidity"]) &
                             (merged_df["Zone Air Relative Humidity"] <= cluster_range["max_humidity"]) ]
    
    X = cluster_data[["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"]]
    y = cluster_data["Zone Air CO2 Concentration"]
    
    model = LinearRegression()
    model.fit(X, y)
    return model

def validate_sensor_data(lr_model, sensor_data,assembler):
    subfeatures_df = sensor_data[["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"]]
    assembled_sensor_data = assembler.transform(subfeatures_df)
    X = pd.DataFrame(assembled_sensor_data, columns=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"])  
    predictions = lr_model.predict(X)
    
    sensor_data["prediction"] = predictions
    sensor_data["error"] = sensor_data["Zone Air CO2 Concentration"] - predictions
    return sensor_data


def process_sensor_data(sensor_data, kmeans_model, scaler, assembler, cluster_ranges, merged_df):
    sensor_cluster_id, scaled_sensor_data = assign_to_cluster(kmeans_model, scaler, assembler, sensor_data)
    print(f"Sensor data is assigned to cluster: {sensor_cluster_id}")
    #print(scaler.inverse_transform(scaled_sensor_data))
    lr_model = train_regression_for_cluster(sensor_cluster_id, cluster_ranges, merged_df)
    assembler_validate = ColumnTransformer(
        transformers=[
            ("features", FunctionTransformer(lambda x: x, validate=False), 
            ["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"]) 
        ]
    )
    assembler_validate.fit(merged_df)
    validation_results = validate_sensor_data(lr_model, sensor_data, assembler_validate)
    
    return validation_results

# Train a regression model for a given cluster
def train_regression_for_cluster(cluster_id,cluster_ranges,merged_df):
    cluster_data = merged_df[(merged_df["Zone Mean Air Temperature"] >= cluster_ranges.loc[cluster_id, "min_temperature"]) &
                             (merged_df["Zone Mean Air Temperature"] <= cluster_ranges.loc[cluster_id, "max_temperature"]) &
                             (merged_df["_volume"] >= cluster_ranges.loc[cluster_id, "min_volume"]) &
                             (merged_df["_volume"] <= cluster_ranges.loc[cluster_id, "max_volume"]) &
                             (merged_df["Ventilation"] >= cluster_ranges.loc[cluster_id, "min_ventilation"]) &
                             (merged_df["Ventilation"] <= cluster_ranges.loc[cluster_id, "max_ventilation"]) &
                             (merged_df["Zone Air Relative Humidity"] >= cluster_ranges.loc[cluster_id, "min_humidity"]) &
                             (merged_df["Zone Air Relative Humidity"] <= cluster_ranges.loc[cluster_id, "max_humidity"]) ]
    
    X = cluster_data[["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"]]
    y = cluster_data["Zone Air CO2 Concentration"]
    model = LinearRegression().fit(X, y)
    return model

# Load data from CSV
input_file_path = "../decay-simulated/CO2_decay_filtered.csv"
data_df = pd.read_csv(input_file_path, delimiter=";")
# Feature selection and rounding
merged_df = data_df[[
    "Zone Mean Air Temperature", "Zone Air CO2 Concentration", "Zone Air Relative Humidity",
    "_volume", "Ventilation"
]].round(2)

# Calculate variance
variance_df = merged_df.groupby([
    "Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"
]).agg(
    CO2_variance=("Zone Air CO2 Concentration", "var")
).dropna().reset_index()

feature_columns = ["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "CO2_variance"]
variance_df = variance_df.astype(float)
# Define a column transformer to concatenate the selected features
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

scaled_df = assembled_df
scaled_df["scaledFeatures"] = list(scaled_features)

# PCA and hierarchical clustering
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
reduced_data = pca.fit_transform(scaled_features)
linkage_matrix = linkage(reduced_data, method='ward')

plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix, truncate_mode="level", p=5)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.savefig("dendrogram.png")
print("Dendrogram saved")

# Apply KMeans
# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_model = kmeans.fit(scaled_features)
clustered_df = scaled_df
clustered_df["prediction"] = kmeans_model.predict(scaled_features)


# Select specific columns
final_df = clustered_df[["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "CO2_variance", "prediction"]]
final_df = final_df[(final_df["CO2_variance"] > 0) &
                    (final_df["CO2_variance"] < 20) &
                    (final_df["Zone Mean Air Temperature"] > 20) &
                    (final_df["Zone Mean Air Temperature"] < 40) &
                    (final_df["Zone Air Relative Humidity"] > 20) &
                    (final_df["Zone Air Relative Humidity"] < 80) &
                    (final_df["_volume"] > 55) &
                    (final_df["_volume"] < 75)]


# Extract cluster ranges
cluster_ranges = final_df.groupby("prediction").agg(
    min_temperature=("Zone Mean Air Temperature", "min"),
    max_temperature=("Zone Mean Air Temperature", "max"),
    min_volume=("_volume", "min"),
    max_volume=("_volume", "max"),
    min_ventilation=("Ventilation", "min"),
    max_ventilation=("Ventilation", "max"),
    min_humidity=("Zone Air Relative Humidity", "min"),
    max_humidity=("Zone Air Relative Humidity", "max")
).reset_index()

print("cluster_ranges")
print(cluster_ranges)

# Train regression model for each cluster
#models = {cluster_id: train_regression_for_cluster(cluster_id) for cluster_id in cluster_ranges.index}
output_path = "output_low_variance_clusters.csv"
# Save clustered data to CSV
variance_df.to_csv(output_path, index=False)

sensor_data_1 = pd.DataFrame([
    {"Zone Mean Air Temperature": 22.35, "Zone Air Relative Humidity": 43.5, "Ventilation": 0.25, "_volume": 65.0, "Zone Air CO2 Concentration": 1150.0}
])
print(sensor_data_1)
validation_results_1 = process_sensor_data(
    sensor_data=sensor_data_1,
    kmeans_model=kmeans_model,
    scaler=scaler_model,
    assembler=assembler,
    cluster_ranges=cluster_ranges,
    merged_df=merged_df
)

print(validation_results_1)

sensor_data_2 = pd.DataFrame([
    {"Zone Mean Air Temperature": 22.35, "Zone Air Relative Humidity": 43.5, "Ventilation": 0.25, "_volume": 65.0, "Zone Air CO2 Concentration": 5000.0}
])
print(sensor_data_2)

validation_results_2 = process_sensor_data(
    sensor_data=sensor_data_2,
    kmeans_model=kmeans_model,
    scaler=scaler_model,
    assembler=assembler,
    cluster_ranges=cluster_ranges,
    merged_df=merged_df
)

print(validation_results_2)
df = pd.read_csv(output_path)

df = df[[
    "Zone Mean Air Temperature",
    "Zone Air Relative Humidity",
    "_volume",
    "CO2_variance",
    "prediction"
]]

filtered_df = df[df["prediction"] == 3]
print(filtered_df)
count = len(filtered_df)
print(f"Number of rows: {count}")
# Visualization
sns.pairplot(
    df,
    vars=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "_volume", "CO2_variance"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
plt.savefig("cluster_pairplot.png")
print("Pairplot saved")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    variance_df["Zone Mean Air Temperature"],
    variance_df["Zone Air Relative Humidity"],
    variance_df["_volume"],
    c=variance_df["prediction"],
    cmap="tab10",
    s=50
)
ax.set_xlabel("Zone Mean Air Temperature")
ax.set_ylabel("Zone Air Relative Humidity")
ax.set_zlabel("Volume")
ax.set_title("3D Cluster Visualization")
plt.savefig("cluster_3d_plot.png")
print("3D plot saved")
