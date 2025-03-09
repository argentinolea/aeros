from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, lag, when, count, sum as ps_sum, first, last, min as ps_min, max as ps_max, avg, lit, round as ps_round
from pyspark.sql import SparkSession, Window
import pandas as pd
import matplotlib.pyplot as plt
import h5py

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

def process_co2_decay_events(df):
    # Ensure presence is integer (0/1)
    df = df.withColumn("BinaryOccupancy", col("BinaryOccupancy").cast("int"))

    # Identify sequences where presence transitions from 1 → 0 (occupancy drop)
    window_spec = Window.orderBy("Datetime")
    df = df.withColumn("presence_shift", lag("BinaryOccupancy", 1).over(window_spec))
    df = df.withColumn("event_start", when((col("presence_shift") == 1) & (col("BinaryOccupancy") == 0), 1).otherwise(0))

    # Generate cumulative sum to assign event IDs
    df = df.withColumn("event_id", ps_sum("event_start").over(window_spec))
    df = df.withColumn("event_id", when(col("BinaryOccupancy") == 1, None).otherwise(col("event_id")))

    # Filter for decay events (only presence=False sequences)
    decay_events = df.filter(col("event_id").isNotNull())

    # Aggregate data to find CO2 decay trends per event
    event_summary = decay_events.groupBy("event_id").agg(
        first("Datetime").alias("start_time"),
        last("Datetime").alias("end_time"),
        first("Zone Air CO2 Concentration").alias("start_co2"),
        last("Zone Air CO2 Concentration").alias("end_co2"),
        ps_min("Zone Air CO2 Concentration").alias("min_co2"),
        avg("Zone Air CO2 Concentration").alias("co2_trend")
    )

    # Calculate event duration in minutes
    event_summary = event_summary.withColumn(
        "duration_minutes",
        (col("end_time").cast("long") - col("start_time").cast("long")) / 60
    )

    # Apply filtering conditions
    threshold_factor = 1.2  # Allows a 20% increase from min_co2 but rejects anything higher
    filtered_events = event_summary.filter(
        (col("start_co2") > col("end_co2")) &  # Ensuring initial CO2 is higher than final
        (col("min_co2") < col("start_co2")) &  # Ensuring real decay occurred
        (col("end_co2") < threshold_factor * col("min_co2")) &  # Avoid fake decays
        (col("co2_trend") < 0)  # Ensuring CO2 trend is negative (decay)
    )

    # Assign "Ignore" to non-matching presence states
    df = df.withColumn("presence_analysis", col("BinaryOccupancy").cast("string"))
    df = df.withColumn(
        "presence_analysis",
        when((col("BinaryOccupancy") == 0) & (~col("event_id").isin([row["event_id"] for row in filtered_events.collect()])), "Ignore")
        .otherwise(col("presence_analysis"))
    )

    return df, filtered_events

# Clustering function
def cluster_co2_decay_events(filtered_events, n_clusters=3):
    # Prepare feature vector for clustering
    assembler = VectorAssembler(inputCols=["start_co2", "end_co2", "duration_minutes"], outputCol="features")
    filtered_events = assembler.transform(filtered_events)

    # Apply KMeans clustering
    kmeans = KMeans(k=n_clusters, seed=42, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(filtered_events)
    filtered_events = model.transform(filtered_events)

    # Convert to Pandas for plotting
    filtered_events_pd = filtered_events.select("start_co2", "duration_minutes", "cluster").toPandas()

    # Plot clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        filtered_events_pd["start_co2"],
        filtered_events_pd["duration_minutes"],
        c=filtered_events_pd["cluster"],
        cmap="viridis",
        edgecolors='k'
    )
    plt.xlabel("Start CO2")
    plt.ylabel("Duration (minutes)")
    plt.title("Clusters of CO2 Decay Events")
    plt.colorbar(label="Cluster ID")
    plt.show()

    return filtered_events

# Calculate average minimum CO2 value for each cluster
def calculate_average_min_co2(filtered_events):
    return filtered_events.groupby("cluster")["end_co2"].mean()

# Calculate decay constant lambda for each row
def calculate_decay_constant(filtered_events):
    filtered_events = filtered_events.withColumn(
        "decay_constant",
        -log(col("end_co2") / col("start_co2")) / col("duration_minutes")
    )
    return filtered_events

def calculate_decay_constants(df, filtered_events):
    df = df.join(
        filtered_events.select("event_id", "start_time", "end_time", "start_co2", "end_co2"),
        on="event_id",
        how="left"
    )

    df = df.withColumn(
        "measurement_decay_constant",
        -log(col("Zone Air CO2 Concentration") / col("start_co2")) /
        ((col("Datetime").cast("long") - col("start_time").cast("long")) / 60)
    )

    return df

def plot_co2_decay_events(df):
    plt.figure(figsize=(12, 6))
    for event_id, group in df.groupby("event_id"):
        if not group["start_co2"].isna().all():  # Ensure event has valid data
            plt.plot(group["date"], group["co2"], label=f"Event {int(event_id)}", alpha=0.6)
    
    plt.xlabel("Time")
    plt.ylabel("CO2 Concentration (ppm)")
    plt.title("CO2 Decay Events Over Time")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", ncol=1, frameon=False)
    plt.grid(True)
    pairplot_output_path = "cluster_decay_pairplot.png"
    plt.savefig(pairplot_output_path)
    
    
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



df, decay_events = process_co2_decay_events(df)
df.show(10)
clustered_events = cluster_co2_decay_events(decay_events)
avg_min_co2 = calculate_average_min_co2(clustered_events)
clustered_events = calculate_decay_constant(clustered_events)
df_with_constants = calculate_decay_constants(df, clustered_events)

df_with_constants.to_csv("CO2_decay_with_constants.csv",sep=";", index=False)
print("Average Minimum CO2 per Cluster:")
print(avg_min_co2)
print("Decay Constants:")
print(clustered_events[["cluster", "decay_constant"]])
plot_co2_decay_events(df_with_constants)
