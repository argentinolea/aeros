from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import count as ps_count, min as ps_min, max as ps_max, avg as ps_avg, variance, round as ps_round
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import pandas as pd
import h5py

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Python Spark Clustering with Standardization") \
    .master("spark://192.168.1.120:7077") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.task.maxDirectResultSize", "10M") \
    .getOrCreate()

# Path to your HDF5 file
input_file_path = "dataset_office_rooms.h5"

# Function to read and flatten a subset of records based on the metadata
def read_hdf5_subset(file_path, key, start_row=0, num_records=10000000):
    with h5py.File(file_path, 'r') as h5f:
        dataset = h5f[key]
        
        if not hasattr(dataset, "shape"):
            raise ValueError(f"The key '{key}' does not point to a dataset but to a group. Specify the full path to the dataset.")
        
        # Ensure we don't exceed the total number of rows
        total_rows = dataset.shape[0]
        end_row = min(start_row + num_records, total_rows)
        
        # Extract column names
        subset_data = dataset[start_row:end_row]
        
        # Flatten multidimensional fields based on the provided metadata
        flattened_data = {
            "index": subset_data["index"],
            "Zone Air CO2 Concentration": subset_data["values_block_0"][:, 0],
            "Zone Mean Air Temperature": subset_data["values_block_0"][:, 1],
            "Zone Air Relative Humidity": subset_data["values_block_0"][:, 2],
            "Occupancy": subset_data["values_block_0"][:, 3],
            "Ventilation": subset_data["values_block_0"][:, 4],
            "simID": subset_data["values_block_1"][:, 0],
            "BinaryOccupancy": subset_data["values_block_1"][:, 1],
            "Datetime": subset_data["values_block_2"][:, 0].astype(str),  # Convert bytes to string
        }

        # Convert to Pandas DataFrame
        return pd.DataFrame(flattened_data)

# Read a subset of records from the HDF5 file
dataset_key = "data/table"
chunk = read_hdf5_subset(input_file_path, key=dataset_key, start_row=0, num_records=1000000)

# Convert the chunk to Spark DataFrame
spark_df = spark.createDataFrame(chunk)

# Round the features to 2 decimal places
spark_df = spark_df.withColumn("Zone Mean Air Temperature", ps_round(col("Zone Mean Air Temperature"), 2)) \
                   .withColumn("Zone Air CO2 Concentration", ps_round(col("Zone Air CO2 Concentration"), 2)) \
                   .withColumn("Zone Air Relative Humidity", ps_round(col("Zone Air Relative Humidity"), 2))

# Assemble the clustering features
clustering_features = ["Zone Mean Air Temperature", "Zone Air CO2 Concentration", "Zone Air Relative Humidity", "Ventilation"]
assembler = VectorAssembler(inputCols=clustering_features, outputCol="features")
assembled_data = assembler.transform(spark_df)

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

# Perform K-Means Clustering on standardized features
kmeans = KMeans(k=3, seed=1, featuresCol="scaled_features", predictionCol="cluster")
kmeans_model = kmeans.fit(scaled_data)

# Add cluster labels to the DataFrame
clustered_data = kmeans_model.transform(scaled_data)

# Analyze clusters
cluster_stats = clustered_data.groupBy("cluster").agg(
    ps_count("cluster").alias("count"),
    ps_round(ps_avg("Zone Mean Air Temperature"), 2).alias("avg_temperature"),
    ps_round(ps_avg("Zone Air CO2 Concentration"), 2).alias("avg_co2"),
    ps_round(ps_avg("Zone Air Relative Humidity"), 2).alias("avg_humidity"),
    ps_round(ps_avg("Ventilation"), 2).alias("avg_ventilation"),
    ps_round(ps_avg("Occupancy"), 2).alias("avg_occupancy")
)

# Show cluster statistics
print("Cluster Statistics:")
cluster_stats.show()

print("Cluster Centers (Centroids):")
for idx, center in enumerate(kmeans_model.clusterCenters()):
    print(f"Cluster {idx}: {center}")

# Filter data where temperature is approximately 23.9 degrees
filtered_data = spark_df.filter((col("Zone Mean Air Temperature") >= 20.62) & (col("Zone Mean Air Temperature") <= 20.7))
num_records = filtered_data.count()
print(f"Number of records in filtered_data: {num_records}")
filtered_data.show()
output_path = "filtered_20_data.csv"

# Export filtered_data to a CSV file
filtered_data.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

print(f"Filtered data has been exported to {output_path}")
# Group by Occupancy and calculate CO2 variance
individual_variances = filtered_data.groupBy("Occupancy").agg(
    ps_round(variance("Zone Air CO2 Concentration"), 2).alias("co2_variance"),
    ps_round(variance("Zone Mean Air Temperature"), 2).alias("temperature_variance"),
    ps_round(variance("Zone Air Relative Humidity"), 2).alias("humidity_variance"),
    ps_round(variance("Ventilation"), 2).alias("ventilation_variance")
)

combined_variance_stats = individual_variances.withColumn(
    "combined_variance",
    col("co2_variance") + col("temperature_variance") + col("humidity_variance") + col("ventilation_variance")
)

# Show the results
print("CO2 Variance for each Occupancy Level (Temperature ~ 23.9°C):")
combined_variance_stats.show()

# Additional analysis as needed