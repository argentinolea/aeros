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
from pyspark.sql import Row

    
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
row_count = merged_df.count()
print(f"Number of rows: {row_count}")

feature_columns = ["Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume", "Zone Air CO2 Concentration"]
target_col = "#occupants"

# Assemble features
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
assembled_df = assembler.transform(merged_df).select("features", target_col)

# Train/test split
train_df, test_df = assembled_df.randomSplit([0.7, 0.3], seed=42)

# Train model
lr = LinearRegression(featuresCol="features", labelCol=target_col)
model = lr.fit(train_df)

# Predict
predictions = model.transform(test_df)

# Evaluate
evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")
mae = evaluator.evaluate(predictions)

evaluator.setMetricName("mse")
mse = evaluator.evaluate(predictions)

rmse = model.summary.rootMeanSquaredError
r2 = model.summary.r2

if r2 < 0:
    negative_r2_clusters.append(cluster_id)

print(f"   MAE  (Mean Absolute Error)      : {mae:.2f} ppm")
print(f"   MSE  (Mean Squared Error)       : {mse:.2f} ppm²")
print(f"   RMSE (Root Mean Squared Error)  : {rmse:.2f} ppm")
print(f"   R²   (Coefficient of Determination): {r2:.4f}")

sensor_data_1 = spark.createDataFrame([
    Row(temperature=22.35, humidity=43.5, ventilation_rate=0.25, volume=65.0, co2=1150.0)
]).withColumnRenamed("ventilation_rate", "ventilation rate")

sensor_data = spark.createDataFrame([
    {
        "Zone Mean Air Temperature": 20.15,
        "Zone Air Relative Humidity": 26.5,
        "Ventilation": 0.00,
        "_volume": 68.56,
        "Zone Air CO2 Concentration": 1150.0
    }
])


sensor_data_1_vec = assembler.transform(sensor_data).select("features")
predicted = model.transform(sensor_data_1_vec).select("prediction").collect()[0][0]

print(f"Predicted number of occupants: {predicted:.2f}")

pandas_df = merged_df.toPandas()
sns.set_style('dark')

plot = sns.relplot(
    x='#occupants',
    y='Zone Air CO2 Concentration',
    data=pandas_df, 
    height=3.8,
    aspect=1.8,
    kind='scatter'
)

plot.fig.savefig("occupancy_vs_co2.png", dpi=300)
