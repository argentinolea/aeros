from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import to_timestamp, concat,col, date_format, log,trim, unix_timestamp, lag, when, count, sum as ps_sum, first, last, min as ps_min, max as ps_max, avg, lit, round as ps_round
from pyspark.sql.functions import col, lit, unix_timestamp, exp, round, when
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StringType
from pyspark.sql.types import TimestampType,BooleanType,DoubleType
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import shutil

spark = SparkSession.builder \
    .appName("Identify CO2 Low Variance Clusters") \
    .master("spark://192.168.1.120:7077") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.task.maxDirectResultSize", "10M") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

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

@udf(returnType=TimestampType())
def fix_date_time_udf(date_str):
    import pandas as pd
    date_str = date_str.strip()
    full_date_str = f"2024/{date_str}"
    if "24:00:00" in full_date_str:
        date_fixed = full_date_str.replace("24:00:00", "00:00:00")
        dt = pd.to_datetime(date_fixed, format="%Y/%m/%d  %H:%M:%S")
        return dt + pd.Timedelta(days=1)
    else:
        return pd.to_datetime(full_date_str, format="%Y/%m/%d  %H:%M:%S")
    
data_df = read_data_table(input_file_path)
metadata_df = read_metadata_table(input_file_path)

data_spark_df = spark.createDataFrame(data_df)
metadata_spark_df = spark.createDataFrame(metadata_df)

df = data_spark_df.join(metadata_spark_df, on="simID", how="inner")
df.show(10)
df = df.withColumn("Zone Mean Air Temperature", ps_round(col("Zone Mean Air Temperature"), 2)) \
                     .withColumn("Zone Air CO2 Concentration", ps_round(col("Zone Air CO2 Concentration"), 2)) \
                     .withColumn("Zone Air Relative Humidity", ps_round(col("Zone Air Relative Humidity"), 2)) \
                     .withColumn("_volume", ps_round(col("_volume"), 2))
df = df.withColumn("#occupants", ps_round(col("maxOccupants")*col("Occupancy"), 2))
df = df.filter((col("_volume") >= 55) & (col("_volume") <= 75))
df = df.withColumn("Datetime", fix_date_time_udf(col("Datetime")))

df = df.withColumn("BinaryOccupancy", col("BinaryOccupancy").cast(BooleanType()))

# Convert '#occupants' to numeric (float) and set errors to null
df = df.withColumn("#occupants", when(col("#occupants").rlike("^[0-9.]+$"), col("#occupants").cast(DoubleType())).otherwise(None))

# Initialize the exponential decay column
df = df.withColumn("exp_decay_#occupants", col("#occupants"))

# Ventilation rate and decay constant
ventilation_rate = 0.75  # air changes per hour
decay_constant = ventilation_rate / 3600  # per second

# Loop to detect presence transitions and apply decay
df = df.withColumn("exp_decay_#occupants", col("#occupants"))

# Create a window spec to partition by presence and order by time
window_spec = Window.orderBy("Datetime")

# Calculate the decay based on the last known occupancy when presence = True
df = df.withColumn(
    "decay_start_time",
    when(col("BinaryOccupancy") == True, col("Datetime")).otherwise(lit(None))
)

# We need to propagate the last known value of "#occupants" where presence == True
df = df.withColumn(
    "prev_occupants",
    when(col("BinaryOccupancy") == True, col("#occupants")).otherwise(lit(None))
)

df.show(10)
# Fill in the previous value to each row where presence == False
df = df.withColumn(
    "prev_occupants",
    when(col("prev_occupants").isNull(), col("prev_occupants").over(window_spec)).otherwise(col("prev_occupants"))
)

# Calculate the elapsed time in seconds from the previous time
df = df.withColumn(
    "elapsed_sec",
    unix_timestamp("Datetime") - unix_timestamp("decay_start_time")
)

# Apply the exponential decay formula: N(t) = N0 * exp(-decay_constant * elapsed_sec)
df = df.withColumn(
    "exp_decay_#occupants",
    when(col("prev_occupants").isNotNull(), col("prev_occupants") * exp(-decay_constant * col("elapsed_sec")))
    .otherwise(col("exp_decay_#occupants"))
)

# Round the result to 2 decimals and set values close to 0 to exactly 0
df = df.withColumn(
    "exp_decay_#occupants",
    when(col("exp_decay_#occupants") < 0.01, lit(0)).otherwise(round(col("exp_decay_#occupants"), 2))
)

# Show the result
df.select("Datetime", "Occupancy", "#occupants", "exp_decay_#occupants").show(20)

# (Optional) Save the output
df.write.csv("output_with_truncated_exponential_decay.csv", header=True)