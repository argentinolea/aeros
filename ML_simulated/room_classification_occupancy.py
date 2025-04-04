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

merged_df = merged_df.withColumn("Zone Air CO2 Concentration", ps_round(col("Zone Air CO2 Concentration"), 2)) 
merged_df = merged_df.withColumn("#occupants", ps_round(col("maxOccupants")*col("Occupancy"), 2))
merged_df = merged_df.filter((col("_volume") >= 55) & (col("_volume") <= 75))
occupant_assembler = VectorAssembler(inputCols=[
    "Zone Air CO2 Concentration", "#occupants"
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



# Apply KMeans
# Apply KMeans clustering
kmeans = KMeans(featuresCol="scaledFeatures", k=5, seed=42)
kmeans_model = kmeans.fit(scaled_df)
clustered_df = kmeans_model.transform(scaled_df)
print("clustered_df")
clustered_df.show()
#clustered_df["prediction"] = kmeans_model.predict(scaled_features)

final_df = clustered_df.select(
    "Zone Air CO2 Concentration", "#occupants", "prediction"
)


cluster_ranges = final_df.groupBy("prediction").agg(
    ps_min("Zone Air CO2 Concentration").alias("min_co2"),
    ps_max("Zone Air CO2 Concentration").alias("max_co2"),
    ps_min("#occupants").alias("min_#occupants"),
    ps_max("#occupants").alias("max_#occupants")
)

print("cluster_ranges")
print(cluster_ranges)

# Convert Spark DataFrame to Pandas
cluster_plot_df = final_df
cluster_plot_df = final_df.select(
    "Zone Air CO2 Concentration", 
    "#occupants", 
    "prediction"
).dropna().toPandas()

sns.set(style="whitegrid")
pairplot = sns.pairplot(
    cluster_plot_df,
    vars=["Zone Air CO2 Concentration","#occupants"],
    hue="prediction",
    palette="tab10",
    diag_kind="kde"
)
pairplot.fig.suptitle("Cluster Visualization by Occupancy", y=1.02)

# Save the plot
pairplot.savefig("cluster_pairplot_occupants.png")
plt.show()
