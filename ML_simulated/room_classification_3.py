from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import variance, col, round as ps_round, lit, collect_set, min as ps_min, max as ps_max
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import h5py
from scipy.cluster.hierarchy import linkage, dendrogram
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import abs as spark_abs, col, mean, stddev, when

def train_regression_models_by_cluster(clustered_df):
    cluster_models = {}
    cluster_metrics = {}
    negative_r2_clusters = []

    feature_cols = ["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    # Loop over each cluster ID
    for cluster_id in clustered_df.select("prediction").distinct().rdd.flatMap(lambda x: x).collect():
        print(f"\n\U0001F52C Training Linear Regression for Cluster {cluster_id}")
        cluster_data = clustered_df.filter(col("prediction") == cluster_id)
        if "features" in cluster_data.columns:
            cluster_data = cluster_data.drop("features")
            
        cluster_data = assembler.transform(cluster_data)
        train_df, test_df = cluster_data.randomSplit([0.7, 0.3], seed=42)

        lr = LinearRegression(featuresCol="features", labelCol="Zone Air CO2 Concentration", predictionCol="lr_prediction")
        model = lr.fit(train_df)

        predictions = model.transform(test_df)

        evaluator = RegressionEvaluator(
            labelCol="Zone Air CO2 Concentration",
            predictionCol="lr_prediction"
        )

        mae = evaluator.setMetricName("mae").evaluate(predictions)
        mse = evaluator.setMetricName("mse").evaluate(predictions)
        rmse = evaluator.setMetricName("rmse").evaluate(predictions)
        r2 = evaluator.setMetricName("r2").evaluate(predictions)
        if(r2 < 0):
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

def validate_sensor_against_all_clusters(sensor_data, cluster_ranges, cluster_models, sensor_assembler):
    sensor_row = sensor_data.first()
    t = sensor_row["Zone Mean Air Temperature"]
    h = sensor_row["Zone Air Relative Humidity"]
    v = sensor_row["Ventilation"]
    vol = sensor_row["_volume"]
    co2 = sensor_row["Zone Air CO2 Concentration"]

    matched_clusters = []

    for row in cluster_ranges.collect():
        cluster_id = row["prediction"]

        if (
            row["min_temperature"] <= t <= row["max_temperature"] and
            row["min_humidity"] <= h <= row["max_humidity"] and
            row["min_ventilation"] <= v <= row["max_ventilation"] and
            row["min_volume"] <= vol <= row["max_volume"]
        ):
            model = cluster_models.get(cluster_id)
            if model:
                assembled = sensor_assembler.transform(sensor_data)
                predicted = model.setPredictionCol("lr_prediction").transform(assembled)

                predicted = predicted.withColumn(
                    "error", spark_abs(col("Zone Air CO2 Concentration") - col("lr_prediction"))
                ).withColumn("cluster_id", lit(cluster_id))

                matched_clusters.append(predicted.select(
                    "Zone Air CO2 Concentration", "lr_prediction", "error", "cluster_id"
                ).first())

    if not matched_clusters:
        print("❌ No matching cluster found for sensor data.")
        return

    print("✅ Sensor matched the following clusters:")
    for match in matched_clusters:
        if match["lr_prediction"] < 0:
            continue 
        min_occ = cluster_ranges.filter(col("prediction") == match['cluster_id']).select("min_#occupants").first()["min_#occupants"]
        max_occ = cluster_ranges.filter(col("prediction") == match['cluster_id']).select("max_#occupants").first()["max_#occupants"]    
        print(f"  Cluster {match['cluster_id']}:")
        print(f"     CO₂ actual   : {match['Zone Air CO2 Concentration']:.2f}")
        print(f"     CO₂ predicted: {match['lr_prediction']:.2f}")
        print(f"     Error        : {match['error']:.2f}")
        print(f"     min_#occupants: {min_occ}")
        print(f"     max_#occupants: {max_occ}")

spark = SparkSession.builder \
    .appName("Identify CO2 Clusters") \
    .master("spark://192.168.1.120:7077") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.task.maxDirectResultSize", "10M") \
    .getOrCreate()

input_file_path = "../coddora/dataset_office_rooms.h5"

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
merged_df = merged_df.withColumn("#occupants", ps_round(col("maxOccupants")*col("Occupancy"), 2))
       
                    
print(merged_df.head(10))

merged_df = merged_df.filter(
    (col("Zone Air CO2 Concentration") > 300) & 
    (col("Zone Mean Air Temperature") > 20) & 
    (col("Zone Mean Air Temperature") < 25) & 
    (col("Zone Air Relative Humidity") > 20) & 
    (col("Zone Air Relative Humidity") < 80) & 
    (col("_volume") > 55) & 
    (col("_volume") < 75)
)
occupant_assembler = VectorAssembler(inputCols=[
    "Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "#occupants"
], outputCol="features")
assembled_df = occupant_assembler.transform(merged_df)

print("assembled_df")
assembled_df.show()
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
scaler_model = scaler.fit(assembled_df)
scaled_df = scaler_model.transform(assembled_df)
print("scaled_df")
scaled_df.show()
scaled_features = scaled_df.select("scaledFeatures").rdd.map(lambda row: row.scaledFeatures.toArray()).collect()
scaled_features_df = pd.DataFrame(scaled_features)

kmeans = KMeans(featuresCol="scaledFeatures", k=6, seed=42)
kmeans_model = kmeans.fit(scaled_df)
clustered_df = kmeans_model.transform(scaled_df)
print("clustered_df")
clustered_df.show()
cluster_models, cluster_metrics, negative_r2_clusters = train_regression_models_by_cluster(clustered_df)
clustered_df = clustered_df.filter(~col("prediction").isin(negative_r2_clusters))

final_df = clustered_df.select(
    "Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "#occupants", "prediction"
)

cluster_ranges = final_df.groupBy("prediction").agg(
    ps_min("Zone Mean Air Temperature").alias("min_temperature"),
    ps_max("Zone Mean Air Temperature").alias("max_temperature"),
    ps_min("_volume").alias("min_volume"),
    ps_max("_volume").alias("max_volume"),
    ps_min("Ventilation").alias("min_ventilation"),
    ps_max("Ventilation").alias("max_ventilation"),
    ps_min("Zone Air Relative Humidity").alias("min_humidity"),
    ps_max("Zone Air Relative Humidity").alias("max_humidity"),
    ps_min("#occupants").alias("min_#occupants"),
    ps_max("#occupants").alias("max_#occupants")
)

cluster_ranges.show()

# Convert Spark DataFrame to Pandas
cluster_plot_df = final_df.select(
    "Zone Mean Air Temperature", 
    "Zone Air Relative Humidity", 
    "#occupants", 
    "prediction"
).dropna().toPandas()

sns.set(style="whitegrid")
pairplot = sns.pairplot(
    cluster_plot_df,
    vars=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "#occupants"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("Cluster Visualization by Environmental Features and Occupancy", y=1.02)

# Save the plot
pairplot.savefig("cluster_pairplot_temperature_humidity_occupants.png")
plt.show()


sensor_data = spark.createDataFrame([
    {
        "Zone Mean Air Temperature": 22.5,
        "Zone Air Relative Humidity": 45.0,
        "Ventilation": 0.25,
        "_volume": 65.0,
        "Zone Air CO2 Concentration": 1200.0
    }
])

# Use same assembler used in model training (without #occupants!)
sensor_assembler = VectorAssembler(
    inputCols=["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"],
    outputCol="features"
)

validate_sensor_against_all_clusters(
    sensor_data=sensor_data,
    cluster_ranges=cluster_ranges,
    cluster_models=cluster_models,
    sensor_assembler=sensor_assembler
)

sensor_data = spark.createDataFrame([
    {
        "Zone Mean Air Temperature": 22.5,
        "Zone Air Relative Humidity": 45.0,
        "Ventilation": 0.25,
        "_volume": 65.0,
        "Zone Air CO2 Concentration": 5000.0
    }
])


validate_sensor_against_all_clusters(
    sensor_data=sensor_data,
    cluster_ranges=cluster_ranges,
    cluster_models=cluster_models,
    sensor_assembler=sensor_assembler
)