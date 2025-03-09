from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import variance, col, round as ps_round, lit, collect_set, min as ps_min, max as ps_max
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def read_csv_data(file_path):
    return pd.read_csv(file_path,sep=";")

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
        (col("temperature") >= min_temperature) &
        (col("temperature") <= max_temperature) &
        (col("volume") >= min_volume) &
        (col("volume") <= max_volume) &
        (col("ventilation rate") >= min_ventilation) &
        (col("ventilation rate") <= max_ventilation) &
        (col("humidity") >= min_humidity) &
        (col("humidity") <= max_humidity)
    )
    
    assembler = VectorAssembler(
        inputCols=["temperature", "humidity", "ventilation rate", "volume"],
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
        inputCols=["temperature", "humidity", "ventilation rate", "volume"],
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

input_file_path = "/media/DATA/uni/thesis/co2data/31012002_fixed_all_features.csv"

data_df = read_csv_data(input_file_path)

data_spark_df = spark.createDataFrame(data_df)
data_spark_df.select("co2").distinct().show()

merged_df = data_spark_df.withColumn("temperature", ps_round(col("temperature"), 2)) \
                     .withColumn("co2", ps_round(col("co2"), 2)) \
                     .withColumn("humidity", ps_round(col("humidity"), 2)) \
                     .withColumn("volume", ps_round(col("volume"), 2))
print(f"DataFrame Count: {merged_df.count()}")
variance_df = merged_df.groupBy(
    "temperature", "humidity", "ventilation rate", "volume"
).agg(
    variance("co2").alias("CO2_variance")
).filter(col("CO2_variance").isNotNull())

print(f"Variance DataFrame Count: {variance_df.count()}")

assembler = VectorAssembler(
    inputCols=["temperature", "humidity", "ventilation rate", "volume", "CO2_variance"],
    outputCol="features"
)
assembled_df = assembler.transform(variance_df)

print(f"Assembled DataFrame Count: {assembled_df.count()}")

scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)

kmeans = KMeans(featuresCol="scaledFeatures", k=4, seed=42)
kmeans_model = kmeans.fit(scaled_df)
clustered_df = kmeans_model.transform(scaled_df)

final_df = clustered_df.select(
    "temperature", "humidity", "ventilation rate", "volume", "CO2_variance", "prediction"
)
final_df = final_df.filter(
    (col("CO2_variance") > 0) &
    (col("CO2_variance") < 20) & 
    (col("temperature") > 20) & 
    (col("temperature") < 40) & 
    (col("humidity") > 20) & 
    (col("humidity") < 80) & 
    (col("volume") > 20) & 
    (col("volume") < 300)
)

final_df.write.csv("output_low_variance_clusters_csv.csv", mode="overwrite", header=True)

cluster_ranges = final_df.groupBy("prediction").agg(
    ps_min("temperature").alias("min_temperature"),
    ps_max("temperature").alias("max_temperature"),
    ps_min("volume").alias("minvolume"),
    ps_max("volume").alias("maxvolume"),
    ps_min("ventilation rate").alias("min_ventilation rate"),
    ps_max("ventilation rate").alias("max_ventilation rate"),
    ps_min("humidity").alias("min_humidity"),
    ps_max("humidity").alias("max_humidity")
)

cluster_ranges.show()

output_path = "output_low_variance_clusters_csv.parquet"
final_df.write.parquet(output_path, mode="overwrite")

distinct_values_df = final_df.groupBy("prediction").agg(
    collect_set("temperature").alias("Distinct Temperatures"),
    collect_set("humidity").alias("Distinct Humidities"),
    collect_set("ventilation rate").alias("Distinct ventilation rate"),
    collect_set("volume").alias("Distinct Volumes"),
    collect_set("CO2_variance").alias("Distinct CO2 Variance")
)

distinct_values_df.show()

sensor_data = spark.createDataFrame([
    {"temperature": 22.5, "humidity": 55.0, "ventilation rate": 0.10, "volume": 50.0, "Zone Air CO2 Concentration": 420.0}
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
    {"temperature": 22.5, "humidity": 55.0, "ventilation rate": 0.10, "volume": 50.0, "Zone Air CO2 Concentration": 1160.0}
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

df = final_df.toPandas()

pairplot = sns.pairplot(
    df,
    vars=["temperature", "humidity", "volume", "CO2_variance"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("Cluster Visualization", y=1.02)

pairplot_output_path = "cluster_pairplot_csv.png"
pairplot.savefig(pairplot_output_path)
print(f"Pairplot saved to {pairplot_output_path}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(
    df["temperature"],
    df["humidity"],
    df["volume"],
    c=df["prediction"],
    cmap="tab10",
    s=50 
)

ax.set_xlabel("Temperature")
ax.set_ylabel("Humidity")
ax.set_zlabel("Volume")
ax.set_title("3D Cluster Visualization")

legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

plot3d_output_path = "cluster_3d_plot_csv.png"
plt.savefig(plot3d_output_path)
print(f"3D plot saved to {plot3d_output_path}")

plt.close(fig)

spark.stop()
