from pyspark.sql.functions import to_timestamp, udf, floor, trim,variance, unix_timestamp, window,collect_list, col, round as ps_round, concat_ws, lit, collect_set, min as ps_min, max as ps_max
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from collections import Counter
import numpy as np

spark = SparkSession.builder \
    .appName("Identify CO2 Low Variance Clusters") \
    .master("spark://192.168.1.120:7077") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.task.maxDirectResultSize", "10M") \
    .getOrCreate()

input_file_path = "../dataset_office_rooms.h5"

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

def shannon_entropy(values):
    """Calculate Shannon entropy from a list of values."""
    counter = Counter(values)
    total = len(values)
    probabilities = np.array([count / total for count in counter.values()])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def shannon_entropy_udf(values):
    if not values:
        return None
    counter = Counter(values)
    total = len(values)
    probabilities = [count / total for count in counter.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return float(entropy)  # Ensure it returns a Python float, not numpy.float64

def process_co2_entropy():
    entropy_results = {}
    for presence_value in ["1", "0"]:
        group_data = df.filter(col("BinaryOccupancy") == presence_value).select("Zone Air CO2 Concentration").rdd.flatMap(lambda x: x).collect()
        if group_data:
            entropy_results[presence_value] = shannon_entropy(group_data)
    
    return entropy_results


def process_co2_entropy_by_day(df,day):  
    df = df.withColumn("BinaryOccupancy", col("BinaryOccupancy").cast("int"))
    df = df.withColumn("Datetime", trim(col("Datetime")))
    df = df.filter(col("Datetime").substr(1, 5) == day)

    df = df.withColumn("FullDatetime", concat_ws(" ", lit("2024"), trim(col("Datetime"))))
    df = df.withColumn("TimeWindow", 
                    (floor(unix_timestamp(col("FullDatetime"), "yyyy MM/dd  HH:mm:ss") / 600) * 60)
                    )
    #df_debug = df.filter(col("BinaryOccupancy") == '1') 
    # Debugging: Print some results
    #df_debug.select("Zone Air CO2 Concentration", "FullDatetime", "TimeWindow", "BinaryOccupancy").show(10, truncate=False)
    df = df.orderBy("Datetime")

    entropy_udf = udf(shannon_entropy_udf, DoubleType())

    # Group by BinaryOccupancy and collect CO2 values
    df_grouped = df.groupBy(col("TimeWindow"), col("BinaryOccupancy")) \
                   .agg(collect_list("Zone Air CO2 Concentration").alias("co2_values"))

    # Apply the entropy function
    df_entropy = df_grouped.withColumn("entropy", entropy_udf(col("co2_values")))

    # Collect results
    results = df_entropy.select("TimeWindow", "BinaryOccupancy", "entropy").collect()

    # Convert results to dictionary format
    timestamps = []
    entropy_results = {"1": [], "0": []}  # Presence as binary

    for row in results:
        timestamps.append(row["TimeWindow"])
        entropy_results[str(row["BinaryOccupancy"])].append(row["entropy"])
    
    return timestamps, entropy_results

def plot_entropy(timestamps, entropy_results):
    """Plot continuous entropy trends over time for presence and absence on 26/11/2024."""
    
    # Ensure entropy_results['1'] and entropy_results['0'] are lists of the same length as timestamps
    entropy_results['1'] = entropy_results.get('1', [])
    entropy_results['0'] = entropy_results.get('0', [])
    
    if isinstance(entropy_results['1'], float):
        entropy_results['1'] = [entropy_results['1']]
    if isinstance(entropy_results['0'], float):
        entropy_results['0'] = [entropy_results['0']]

    # Fill missing values with None to align with timestamps
    max_length = len(timestamps)
    
    # Ensure both lists are of equal length
    entropy_results['1'] += [None] * (max_length - len(entropy_results['1']))
    entropy_results['0'] += [None] * (max_length - len(entropy_results['0']))

    plt.figure(figsize=(12, 6))
    
    plt.plot(timestamps, entropy_results['1'], label='Presence=True', linestyle='-', marker='', linewidth=2)
    plt.plot(timestamps, entropy_results['0'], label='Presence=False', linestyle='-', marker='', linewidth=2)
    
    plt.xlabel('Time')
    plt.ylabel('Shannon Entropy')
    plt.title('Shannon Entropy of CO₂ Levels on 26/11/2024')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    
    # Save the plot
    pairplot_output_path = "entropy.png"
    plt.savefig(pairplot_output_path)

    # Show the plot
    plt.show()
    
data_df = read_data_table(input_file_path)
metadata_df = read_metadata_table(input_file_path)

data_spark_df = spark.createDataFrame(data_df)
metadata_spark_df = spark.createDataFrame(metadata_df)

df = data_spark_df.join(metadata_spark_df, on="simID", how="inner")

df = df.withColumn("Zone Mean Air Temperature", ps_round(col("Zone Mean Air Temperature"), 2)) \
                     .withColumn("Zone Air CO2 Concentration", ps_round(col("Zone Air CO2 Concentration"), 2)) \
                     .withColumn("Zone Air Relative Humidity", ps_round(col("Zone Air Relative Humidity"), 2)) \
                     .withColumn("_volume", ps_round(col("_volume"), 2))
       
df = df.filter((col("_volume") >= 60) & (col("_volume") <= 65))

variance_df = df.groupBy(
    "Zone Mean Air Temperature", "Zone Air Relative Humidity", "Ventilation", "_volume"
).agg(
    variance("Zone Air CO2 Concentration").alias("CO2_variance")
).filter(col("CO2_variance").isNotNull()) 


entropy_values = process_co2_entropy()
    
print("Shannon Entropy of CO₂ levels:")
print(f"Presence=True: {entropy_values.get('1', 'No data')}")
print(f"Presence=False: {entropy_values.get('0', 'No data')}")
timestamps, entropy_values_plot = process_co2_entropy_by_day(df,"11/26")
print(entropy_values_plot)
plot_entropy(timestamps, entropy_values_plot)