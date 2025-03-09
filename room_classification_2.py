from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import variance, col, round as ps_round, lit, collect_set, min as ps_min, max as ps_max
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import h5py


def assign_to_cluster(kmeans_model, scaler_model, clustering_assembler, sensor_data):
    sensor_data = sensor_data.withColumn("CO2_variance", lit(0.0)) 
    assembled_sensor_data = clustering_assembler.transform(sensor_data)
    scaled_sensor_data = scaler_model.transform(assembled_sensor_data)
    cluster = kmeans_model.transform(scaled_sensor_data)
    return cluster.select("prediction").first()["prediction"], scaled_sensor_data


def train_regression_for_cluster(cluster_id, cluster_ranges, merged_df):
    cluster_range = cluster_ranges.filter(col("prediction") == cluster_id).first()
    min_temperature = cluster_range["min_temperature"]
    max_temperature = cluster_range["max_temperature"]
    min_volume = cluster_range["min_volume"]
    max_volume = cluster_range["max_volume"]
    min_ventilation = cluster_range["min_ventilation"]
    max_ventilation = cluster_range["max_ventilation"]
    min_humidity = cluster_range["min_humidity"]
    max_humidity = cluster_range["max_humidity"]

    cluster_data = merged_df.filter(
        (col("Zone Mean Air Temperature") >= min_temperature) &
        (col("Zone Mean Air Temperature") <= max_temperature) &
        (col("_volume") >= min_volume) &
        (col("_volume") <= max_volume) &
        (col("Ventilation") >= min_ventilation) &
        (col("Ventilation") <= max_ventilation) &
        (col("Zone Air Relative Humidity") >= min_humidity) &
        (col("Zone Air Relative Humidity") <= max_humidity)
    )
    
    assembler = VectorAssembler(
        inputCols=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"],
        outputCol="features"
    )
    assembled_data = assembler.transform(cluster_data)

    lr = LinearRegression(featuresCol="features", labelCol="Zone Air CO2 Concentration", regParam=0.1)
    lr_model = lr.fit(assembled_data)

    return lr_model

def validate_sensor_data(lr_model, sensor_data):
    # Check if "features" column exists and rename it to avoid conflicts
    if "features" in sensor_data.columns:
        sensor_data = sensor_data.withColumnRenamed("features", "existing_features")

    assembler = VectorAssembler(
        inputCols=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"],
        outputCol="features"
    )
    assembled_sensor_data = assembler.transform(sensor_data)

    predictions = lr_model.transform(assembled_sensor_data)

    validation_results = predictions.withColumn(
        "error", col("Zone Air CO2 Concentration") - col("prediction")
    )
    return validation_results

def process_sensor_data(sensor_data, kmeans_model, scaler_model, assembler, cluster_ranges, merged_df):
    sensor_cluster_id, scaled_sensor_data = assign_to_cluster(kmeans_model, scaler_model, assembler, sensor_data)
    print(f"Sensor data is assigned to cluster: {sensor_cluster_id}")
    lr_model = train_regression_for_cluster(sensor_cluster_id, cluster_ranges, merged_df)
    validation_results = validate_sensor_data(lr_model, scaled_sensor_data)
    
    return validation_results

spark = SparkSession.builder \
    .appName("Identify CO2 Low Variance Clusters") \
    .master("spark://192.168.1.120:7077") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.task.maxDirectResultSize", "10M") \
    .getOrCreate()

input_file_path = "dataset_office_rooms.h5"

def read_data_table(file_path, key="data/table", start_row=0, num_records=1000000):
    with h5py.File(file_path, 'r') as h5f:
        dataset = h5f[key]
        rows = dataset[start_row:start_row + num_records]
        flattened_data = {
            "index": rows["index"],
            "Zone Air CO2 Concentration": rows["values_block_0"][:, 0],
            "Zone Mean Air Temperature": rows["values_block_0"][:, 1],
            "Zone Air Relative Humidity": rows["values_block_0"][:, 2],
            "Occupancy": rows["values_block_0"][:, 3],
            "Ventilation": rows["values_block_0"][:, 4],
            "simID": rows["values_block_1"][:, 0],
            "BinaryOccupancy": rows["values_block_1"][:, 1],
            "Datetime": rows["values_block_2"][:, 0].astype(str),
        }
        return pd.DataFrame(flattened_data)


def read_metadata_table(file_path, key="metadata/table"):
    with h5py.File(file_path, 'r') as h5f:
        dataset = h5f[key]
        rows = dataset[:]
        flattened_metadata = {
            "maxOccupants": rows["values_block_0"][:, 6],
            "_volume": rows["values_block_0"][:, 8],
            "simID": rows["values_block_1"][:, 1],  # Extract simID
        }
        return pd.DataFrame(flattened_metadata)

data_df = read_data_table(input_file_path)
metadata_df = read_metadata_table(input_file_path)

data_spark_df = spark.createDataFrame(data_df)
metadata_spark_df = spark.createDataFrame(metadata_df)

merged_df = data_spark_df.join(metadata_spark_df, on="simID", how="inner")

merged_df = merged_df.withColumn("Zone Mean Air Temperature", ps_round(col("Zone Mean Air Temperature"), 2)) \
                     .withColumn("Zone Air CO2 Concentration", ps_round(col("Zone Air CO2 Concentration"), 2)) \
                     .withColumn("Zone Air Relative Humidity", ps_round(col("Zone Air Relative Humidity"), 2)) \
                     .withColumn("_volume", ps_round(col("_volume"), 2))

variance_df = merged_df.groupBy(
    "Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"
).agg(
    variance("Zone Air CO2 Concentration").alias("CO2_variance")
).filter(col("CO2_variance").isNotNull()) 

assembler = VectorAssembler(
    inputCols=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "CO2_variance"],
    outputCol="features"
)
assembled_df = assembler.transform(variance_df)

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

kmeans = KMeans(featuresCol="scaledFeatures", k=4, seed=42)
kmeans_model = kmeans.fit(scaled_df)
clustered_df = kmeans_model.transform(scaled_df)

final_df = clustered_df.select(
    "Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "CO2_variance", "prediction"
)
final_df = final_df.filter(
    (col("CO2_variance") > 0) &
    (col("CO2_variance") < 20) & 
    (col("Zone Mean Air Temperature") > 20) & 
    (col("Zone Mean Air Temperature") < 40) & 
    (col("Zone Air Relative Humidity") > 20) & 
    (col("Zone Air Relative Humidity") < 80) & 
    (col("_volume") > 20) & 
    (col("_volume") < 300)
)

cluster_ranges = final_df.groupBy("prediction").agg(
    ps_min("Zone Mean Air Temperature").alias("min_temperature"),
    ps_max("Zone Mean Air Temperature").alias("max_temperature"),
    ps_min("_volume").alias("min_volume"),
    ps_max("_volume").alias("max_volume"),
    ps_min("Ventilation").alias("min_ventilation"),
    ps_max("Ventilation").alias("max_ventilation"),
    ps_min("Zone Air Relative Humidity").alias("min_humidity"),
    ps_max("Zone Air Relative Humidity").alias("max_humidity")
)

cluster_ranges.show()

output_path = "output_low_variance_clusters.parquet"
final_df.write.parquet(output_path, mode="overwrite")

distinct_values_df = final_df.groupBy("prediction").agg(
    collect_set("Zone Mean Air Temperature").alias("Distinct Temperatures"),
    collect_set("Zone Air Relative Humidity").alias("Distinct Humidities"),
    collect_set("Ventilation").alias("Distinct Ventilation"),
    collect_set("_volume").alias("Distinct Volumes"),
    collect_set("CO2_variance").alias("Distinct CO2 Variance")
)

distinct_values_df.show()

sensor_data = spark.createDataFrame([
    {"Zone Mean Air Temperature": 22.5, "Zone Air Relative Humidity": 55.0, "Ventilation": 0.10, "_volume": 50.0, "Zone Air CO2 Concentration": 420.0}
])

validation_results = process_sensor_data(
    sensor_data=sensor_data,
    kmeans_model=kmeans_model,
    scaler_model=scaler_model,
    assembler=assembler,
    cluster_ranges=cluster_ranges,
    merged_df=merged_df
)

validation_results.show()

sensor_data = spark.createDataFrame([
    {"Zone Mean Air Temperature": 22.5, "Zone Air Relative Humidity": 55.0, "Ventilation": 0.10, "_volume": 50.0, "Zone Air CO2 Concentration": 1160.0}
])


validation_results = process_sensor_data(
    sensor_data=sensor_data,
    kmeans_model=kmeans_model,
    scaler_model=scaler_model,
    assembler=assembler,
    cluster_ranges=cluster_ranges,
    merged_df=merged_df
)

validation_results.show()

spark.stop()

df = pd.read_parquet(output_path)

df = df[[
    "Zone Mean Air Temperature",
    "Zone Air Relative Humidity",
    "_volume",
    "CO2_variance",
    "prediction"
]]


pairplot = sns.pairplot(
    df,
    vars=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "_volume", "CO2_variance"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("Cluster Visualization", y=1.02)


pairplot_output_path = "cluster_pairplot.png"
pairplot.savefig(pairplot_output_path)
print(f"Pairplot saved to {pairplot_output_path}")


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')


scatter = ax.scatter(
    df["Zone Mean Air Temperature"],
    df["Zone Air Relative Humidity"],
    df["_volume"],
    c=df["prediction"],
    cmap="tab10",
    s=50 
)

ax.set_xlabel("Zone Mean Air Temperature")
ax.set_ylabel("Zone Air Relative Humidity")
ax.set_zlabel("Volume")
ax.set_title("3D Cluster Visualization")

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

plot3d_output_path = "cluster_3d_plot.png"
plt.savefig(plot3d_output_path)
print(f"3D plot saved to {plot3d_output_path}")

plt.close(fig)