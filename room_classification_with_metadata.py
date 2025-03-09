from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql.functions import count as ps_count, min as ps_min, max as ps_max, avg as ps_avg, variance, round as ps_round
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.ml.feature import Bucketizer
import pandas as pd
import h5py

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Python Spark Clustering with Volume") \
    .master("spark://192.168.1.120:7077") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.task.maxDirectResultSize", "10M") \
    .getOrCreate()

# Path to your HDF5 file
input_file_path = "dataset_office_rooms.h5"

# Function to read and flatten data/table
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

# Function to read and flatten metadata/table
def read_metadata_table(file_path, key="metadata/table"):
    with h5py.File(file_path, 'r') as h5f:
        dataset = h5f[key]
        rows = dataset[:]
        flattened_metadata = {
            #"width": rows["values_block_0"][:, 0],
            #"length": rows["values_block_0"][:, 1],
            #"height": rows["values_block_0"][:, 2],
            #"infiltration": rows["values_block_0"][:, 3],
            #"outdoor_co2": rows["values_block_0"][:, 4],
            #"orientation": rows["values_block_0"][:, 5],
            "maxOccupants": rows["values_block_0"][:, 6],
            #"_floorArea": rows["values_block_0"][:, 7],
            "_volume": rows["values_block_0"][:, 8],
            #"exteriorSurfaceArea": rows["values_block_0"][:, 9],
            #"winToFloorRatio": rows["values_block_0"][:, 10],
            "simID": rows["values_block_1"][:, 1],  # Extract simID
        }
        return pd.DataFrame(flattened_metadata)

# Read data and metadata
data_df = read_data_table(input_file_path)
metadata_df = read_metadata_table(input_file_path)

# Convert both to Spark DataFrames
data_spark_df = spark.createDataFrame(data_df)
metadata_spark_df = spark.createDataFrame(metadata_df)

# Join on simID to align rows
merged_df = data_spark_df.join(metadata_spark_df, on="simID", how="inner")

# Round the features to 2 decimal places
merged_df = merged_df.withColumn("Zone Mean Air Temperature", ps_round(col("Zone Mean Air Temperature"), 2)) \
                     .withColumn("Zone Air CO2 Concentration", ps_round(col("Zone Air CO2 Concentration"), 2)) \
                     .withColumn("Zone Air Relative Humidity", ps_round(col("Zone Air Relative Humidity"), 2)) \
                     .withColumn("_volume", ps_round(col("_volume"), 2))

# Assemble the clustering features, including `_volume`
clustering_features = ["Zone Mean Air Temperature", "Zone Air CO2 Concentration", "Zone Air Relative Humidity", "Ventilation", "_volume"]
assembler = VectorAssembler(inputCols=clustering_features, outputCol="features")
assembled_data = assembler.transform(merged_df)

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(assembled_data)
scaled_data = scaler_model.transform(assembled_data)

# Perform K-Means Clustering on standardized features
# Define bins for each feature
temperature_bins = [-float("inf"), 20, 23, 25, 27, float("inf")]
humidity_bins = [-float("inf"), 30, 40, 50, 60, float("inf")]
ventilation_bins = [-float("inf"), 0.1, 0.5, 1.0, 2.0, float("inf")]
volume_bins = [-float("inf"), 50, 100, 200, 300, float("inf")]

# Add intervals for each feature using Bucketizer
bucketizers = {
    "Zone Mean Air Temperature": Bucketizer(
        splits=temperature_bins,
        inputCol="Zone Mean Air Temperature",
        outputCol="temp_interval"
    ),
    "Zone Air Relative Humidity": Bucketizer(
        splits=humidity_bins,
        inputCol="Zone Air Relative Humidity",
        outputCol="humidity_interval"
    ),
    "Ventilation": Bucketizer(
        splits=ventilation_bins,
        inputCol="Ventilation",
        outputCol="ventilation_interval"
    ),
    "_volume": Bucketizer(
        splits=volume_bins,
        inputCol="_volume",
        outputCol="volume_interval"
    )
}

# Apply bucketizers
data_with_intervals = merged_df
for feature, bucketizer in bucketizers.items():
    data_with_intervals = bucketizer.transform(data_with_intervals)

# Group by intervals and calculate CO2 variance
interval_stats = data_with_intervals.groupBy(
    "temp_interval", "humidity_interval", "ventilation_interval", "volume_interval"
).agg(
    variance("Zone Air CO2 Concentration").alias("co2_variance")
).filter(
    col("co2_variance").isNotNull()  # Exclude rows with NULL CO2 variance
).orderBy("co2_variance")

# Show top 5 intervals with the lowest CO2 variance
top_5_intervals = interval_stats.limit(5)
print("Top 5 Intervals with Lowest CO2 Variance:")
top_5_intervals.show()

# Detailed rows for each interval in the top 5
for row in top_5_intervals.collect():
    temp_interval = row["temp_interval"]
    humidity_interval = row["humidity_interval"]
    ventilation_interval = row["ventilation_interval"]
    volume_interval = row["volume_interval"]

    print(f"Details for Interval Combination: "
          f"Temp[{temp_interval}], Humidity[{humidity_interval}], Ventilation[{ventilation_interval}], Volume[{volume_interval}]")
    filtered_data = data_with_intervals.filter(
        (col("temp_interval") == temp_interval) &
        (col("humidity_interval") == humidity_interval) &
        (col("ventilation_interval") == ventilation_interval) &
        (col("volume_interval") == volume_interval)
    )
    filtered_data.show()
    
# Filter data where temperature is approximately 23 degrees
filtered_data = scaled_data.filter((col("Zone Mean Air Temperature") >= 22) & (col("Zone Mean Air Temperature") <= 24) &\
    (col("_volume") >= 77) & (col("_volume") <= 80) &\
      (col("Zone Air Relative Humidity") >= 34.0) & (col("Zone Air Relative Humidity") <= 36)  )
num_records = filtered_data.count()
print(f"Number of records in filtered_data: {num_records}")
#filtered_data.show()
# Drop the unsupported columns before saving
filtered_data_to_save = filtered_data.drop("features", "scaled_features")

# Define output path
output_path = "filtered_20_data.csv"

# Export filtered_data to a single CSV file
filtered_data_to_save.coalesce(1).write.csv(output_path, header=True, mode="overwrite")

print(f"Filtered data has been exported to {output_path}")

# Group by Occupancy and calculate CO2 variance
individual_variances = filtered_data.groupBy("Occupancy").agg(
    ps_round(variance("Zone Air CO2 Concentration"), 2).alias("co2_variance"),
    ps_round(variance("Zone Mean Air Temperature"), 2).alias("temperature_variance"),
    ps_round(variance("Zone Air Relative Humidity"), 2).alias("humidity_variance"),
    ps_round(variance("Ventilation"), 2).alias("ventilation_variance"),
    ps_round(variance("_volume"), 2).alias("volume_variance")
)

combined_variance_stats = individual_variances.withColumn(
    "combined_variance",
    col("co2_variance") + col("temperature_variance") + col("humidity_variance") + col("ventilation_variance")+ col("volume_variance")
)

# Show the results
print("CO2 Variance for each Occupancy Level (Temperature ~ 22-24°C):")
combined_variance_stats.show()


# Filter data where temperature is approximately 25 degrees
filtered_data = scaled_data.filter((col("Zone Mean Air Temperature") >= 24) & (col("Zone Mean Air Temperature") <= 26)  &\
    (col("_volume") >= 140) & (col("_volume") <= 145) &\
      (col("Zone Air Relative Humidity") >= 30.0) & (col("Zone Air Relative Humidity") <= 31)  )
num_records = filtered_data.count()
print(f"Number of records in filtered_data: {num_records}")

#filtered_data.show()

# Group by Occupancy and calculate CO2 variance
individual_variances = filtered_data.groupBy("Occupancy").agg(
    ps_round(variance("Zone Air CO2 Concentration"), 2).alias("co2_variance"),
    ps_round(variance("Zone Mean Air Temperature"), 2).alias("temperature_variance"),
    ps_round(variance("Zone Air Relative Humidity"), 2).alias("humidity_variance"),
    ps_round(variance("Ventilation"), 2).alias("ventilation_variance"),
    ps_round(variance("_volume"), 2).alias("volume_variance")
)

combined_variance_stats = individual_variances.withColumn(
    "combined_variance",
    col("co2_variance") + col("temperature_variance") + col("humidity_variance") + col("ventilation_variance")+ col("volume_variance")
)

# Show the results
print("CO2 Variance for each Occupancy Level (Temperature ~ 24-26°C):")
combined_variance_stats.show()


# Filter data where temperature is approximately 25 degrees
filtered_data = scaled_data.filter((col("Zone Mean Air Temperature") >= 28) & (col("Zone Mean Air Temperature") <= 29) &\
    (col("_volume") >= 92) & (col("_volume") <= 93) &\
      (col("Zone Air Relative Humidity") >= 37) & (col("Zone Air Relative Humidity") <= 38)  )
num_records = filtered_data.count()
print(f"Number of records in filtered_data: {num_records}")

#filtered_data.show()

# Group by Occupancy and calculate CO2 variance
individual_variances = filtered_data.groupBy("Occupancy").agg(
    ps_round(variance("Zone Air CO2 Concentration"), 2).alias("co2_variance"),
    ps_round(variance("Zone Mean Air Temperature"), 2).alias("temperature_variance"),
    ps_round(variance("Zone Air Relative Humidity"), 2).alias("humidity_variance"),
    ps_round(variance("Ventilation"), 2).alias("ventilation_variance"),
    ps_round(variance("_volume"), 2).alias("volume_variance")
)

combined_variance_stats = individual_variances.withColumn(
    "combined_variance",
    col("co2_variance") + col("temperature_variance") + col("humidity_variance") + col("ventilation_variance")+ col("volume_variance")
)

# Show the results
print("CO2 Variance for each Occupancy Level (Temperature ~ 29-30°C):")
combined_variance_stats.show()