from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import to_timestamp, concat,col, date_format, log,trim, unix_timestamp, lag, when, count, sum as ps_sum, first, last, min as ps_min, max as ps_max, avg, lit, round as ps_round
from pyspark.sql.functions import col, lit, unix_timestamp, exp, round, when
from pyspark.sql.functions import udf
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StringType,IntegerType
from pyspark.sql.types import TimestampType,BooleanType,DoubleType
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import os
import shutil
import datetime
import glob
import re
import os

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

def override_flag(ts):
    for event_time in event_starts:
        if event_time <= ts < event_time + datetime.timedelta(minutes=40):
            return True
    return False

def get_last_occupants(ts):
    prev = [row for row in presence_data if row["Datetime"] < ts]
    return prev[-1]["#occupants"] if prev else None

data_df = read_data_table(input_file_path)
metadata_df = read_metadata_table(input_file_path)

data_spark_df = spark.createDataFrame(data_df)
metadata_spark_df = spark.createDataFrame(metadata_df)

df = data_spark_df.join(metadata_spark_df, on="simID", how="inner")
df = df.withColumn("Zone Mean Air Temperature", ps_round(col("Zone Mean Air Temperature"), 2)) \
                     .withColumn("Zone Air CO2 Concentration", ps_round(col("Zone Air CO2 Concentration"), 2)) \
                     .withColumn("Zone Air Relative Humidity", ps_round(col("Zone Air Relative Humidity"), 2)) \
                     .withColumn("_volume", ps_round(col("_volume"), 2))
df = df.withColumn("#occupants", round(col("maxOccupants") * col("Occupancy")).cast("int"))
df = df.filter((col("_volume") >= 55) & (col("_volume") <= 75))
df = df.withColumn("Datetime", fix_date_time_udf(col("Datetime")))
row_count = df.count()
print(f"Number of rows: {row_count}")
df = df.withColumn("BinaryOccupancy", col("BinaryOccupancy").cast("int"))
df = df.withColumn("presence", (col("BinaryOccupancy") == 1))
window_spec = Window.orderBy("Datetime")
CO2_occupant_dir = "65m3_export"
df.select(
    "Datetime",
    "Zone Air CO2 Concentration",
    "Zone Mean Air Temperature",
    "Zone Air Relative Humidity",
    "Ventilation",
    "_volume",
    "presence",
    "#occupants",
).coalesce(1).write.mode("overwrite").option("header", False).csv(CO2_occupant_dir)

part_file = glob.glob(os.path.join(CO2_occupant_dir, "part-*.csv"))[0]
final_output_path = "65m3_export.csv"

shutil.move(part_file, final_output_path)

# Step 3: Clean up the temporary directory
shutil.rmtree(CO2_occupant_dir)