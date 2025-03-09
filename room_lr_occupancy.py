from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import count as ps_count, min as ps_min, max as ps_max, avg as ps_avg
import pandas as pd
import h5py


# Initialize Spark session
spark = SparkSession.builder \
    .appName("Python Spark Connection") \
    .master("spark://192.168.1.120:7077") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .getOrCreate()

# Path to your HDF5 file
input_file_path = "dataset_office_rooms.h5"

# Function to read and flatten a subset of records based on the metadata
def read_hdf5_subset(file_path, key, start_row=0, num_records=100000):
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

# Read only 1000 records
dataset_key = "data/table"
chunk = read_hdf5_subset(input_file_path, key=dataset_key, start_row=0, num_records=1000)

# Convert the chunk to Spark DataFrame
spark_df = spark.createDataFrame(chunk)
print("Columns:", spark_df.columns)

# Extract the latest row based on 'index' (or any other ordering column)
latest_row = spark_df.orderBy(col("index").desc()).limit(1)
latest_values = latest_row.select(
    "Zone Air CO2 Concentration", 
    "Occupancy", 
    "Zone Mean Air Temperature", 
    "Zone Air Relative Humidity"
)
print("Latest Values:")
latest_values.show()

# Prepare data for regression
feature_columns = ["Zone Air CO2 Concentration", "Zone Mean Air Temperature", "Zone Air Relative Humidity"]
target_column = "Occupancy"

assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
regression_data = assembler.transform(spark_df).select("features", col(target_column).alias("label"))

# Perform Linear Regression
print("Performing Linear Regression...")
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(regression_data)

# Print Linear Regression model coefficients and intercept
print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")

# Evaluate the model
print("Linear Regression Predictions:")
predictions = lr_model.transform(regression_data)
predictions.select("features", "label", "prediction").show()
stats = predictions.groupBy("label").agg(
    ps_count("prediction").alias("count"),
    ps_avg("prediction").alias("avg_prediction"),
    ps_min("prediction").alias("min_prediction"),
    ps_max("prediction").alias("max_prediction")
)

# Show statistics for each label
stats.show()

# Show distinct features for each label
predictions.select("features", "label").distinct().show()