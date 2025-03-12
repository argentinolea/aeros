import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset
def process_co2_decay_events(file_path):
    df["date"] = pd.to_datetime(df["date"])
    df["presence"] = df["presence"].astype(bool)

    # Identify sequences where presence transitions from True to continuous False
    df["presence_shift"] = df["presence"].shift(1, fill_value=True)
    df["event_start"] = (df["presence_shift"] == True) & (df["presence"] == False)

    # Assign unique event IDs to each decay event
    event_id = df["event_start"].cumsum()
    df["event_id"] = event_id.where(df["presence"] == False, None)

    # Filter out NaN event IDs (only presence=False sequences)
    decay_events = df.dropna(subset=["event_id"])

    # Aggregate data to find the first and last CO2 values per event
    event_summary = decay_events.groupby("event_id").agg(
        start_time=("date", "first"),
        end_time=("date", "last"),
        start_co2=("co2", "first"),
        end_co2=("co2", "last"),
        min_co2=("co2", "min")
    )

    # Calculate duration of each event
    event_summary["duration_minutes"] = (event_summary["end_time"] - event_summary["start_time"]).dt.total_seconds() / 60


   # **Step 1: Split long decay events into segments of 120 minutes**
    min_duration = 30   # Minimum required event duration
    max_duration = 120  # Maximum allowed decay duration

    # Create a new 'segment_id' column based on time window of 120 minutes
    decay_events["time_offset"] = (decay_events["date"] - decay_events.groupby("event_id")["date"].transform("min")).dt.total_seconds() / 60
    decay_events["segment_id"] = (decay_events["time_offset"] // max_duration).astype(int)

    # Aggregate again but with segment-wise breakdown
    segmented_summary = decay_events.groupby(["event_id", "segment_id"]).agg(
        start_time=("date", "first"),
        end_time=("date", "last"),
        start_co2=("co2", "first"),
        end_co2=("co2", "last"),
        min_co2=("co2", "min")
    )
        # Compute CO₂ trend
    segmented_summary["co2_trend"] = decay_events.groupby(["event_id", "segment_id"])["co2"].diff().mean()

    # **Step 2: Calculate segment duration**
    segmented_summary["duration_minutes"] = (segmented_summary["end_time"] - segmented_summary["start_time"]).dt.total_seconds() / 60

    # Define a threshold to exclude fake decay events
    threshold_factor = 1.1  # Allows a 40% increase from min_co2 but rejects anything higher
    
    # Apply updated decay event conditions (Monotonicity in Decay)
    filtered_segments = segmented_summary[
        (segmented_summary["start_co2"] > segmented_summary["end_co2"]) &  # Ensuring initial CO2 is higher than final
        (segmented_summary["min_co2"] < segmented_summary["start_co2"]) &  # Ensuring real decay occurred
        (segmented_summary["end_co2"] < threshold_factor * segmented_summary["min_co2"]) &  # Avoid cases where end CO2 rises too much
        (segmented_summary["co2_trend"] < 0) &  # Ensuring CO2 trend is negative (decay)
        (segmented_summary["duration_minutes"] >= min_duration) &  # Ensuring min duration of 30 min
        (segmented_summary["duration_minutes"] <= max_duration)  # Enforce max duration of 120 min

    ]
    
    # Assign new segment-based event IDs
    filtered_segments["new_event_id"] = filtered_segments.index.map(lambda x: f"{x[0]}_{x[1]}")
    # Create a new column 'presence_analysis'
    df["presence_analysis"] = df["presence"].astype(object)
    #df.loc[(df["presence"] == False) & (~df["event_id"].isin(filtered_events.index)), "presence_analysis"] = True
    df.loc[(df["presence"] == False) & (~df["event_id"].isin(filtered_segments.index.get_level_values(0))), "presence_analysis"] = "Ignore"
    
    print(filtered_segments.head(100))
    return df, filtered_segments

# Clustering function
def cluster_co2_decay_events(filtered_events, n_clusters=3):
    data = filtered_events[["start_co2", "end_co2", "duration_minutes"]]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    filtered_events["cluster"] = kmeans.fit_predict(data)

    # Plot clusters
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        filtered_events["start_co2"],
        filtered_events["duration_minutes"],
        c=filtered_events["cluster"],
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
    filtered_events["decay_constant"] = -np.log(filtered_events["end_co2"] / filtered_events["start_co2"]) / filtered_events["duration_minutes"]
    return filtered_events

def calculate_decay_constants(df, filtered_segments):
    filtered_segments = filtered_segments.reset_index()

    # Verify that 'event_id' and 'segment_id' exist
    if "event_id" not in df.columns:
        raise KeyError("Missing 'event_id' in df.")
    
    if "event_id" not in filtered_segments.columns or "segment_id" not in filtered_segments.columns:
        raise KeyError("Missing 'event_id' or 'segment_id' in filtered_segments.")

    df["event_id"] = df["event_id"].fillna(-1).astype("Int64")  # Use Int64 to allow NaN values
    filtered_segments["event_id"] = filtered_segments["event_id"].fillna(-1).astype("Int64")

    # Ensure df has a 'segment_id' column by recalculating it
    if "segment_id" not in df.columns:
        df["time_offset"] = (df["date"] - df.groupby("event_id")["date"].transform("min")).dt.total_seconds() / 60
        max_duration = 120  # Maximum allowed duration per segment
        df["segment_id"] = (df["time_offset"] // max_duration).astype(int)
    
    # Merge based on both 'event_id' and 'segment_id'
    df = df.merge(
        filtered_segments[["event_id", "segment_id", "start_time", "end_time", "start_co2", "end_co2"]],
        on=["event_id", "segment_id"],  
        how="left"
    )

    time_diff = (df["date"] - df["start_time"]).dt.total_seconds() / 60  # Convert time difference to minutes
    df["measurement_decay_constant"] = np.where(
        (df["co2"] > 0) & (df["start_co2"] > 0) & (time_diff > 0), 
        -np.log(df["co2"] / df["start_co2"]) / time_diff,  
        np.nan  # Assign NaN if values are invalid
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
    
    
# Usage
file_path = "../capture/31010902_fixed_all_features.csv"  # Change to your actual file path
df = pd.read_csv(file_path, delimiter=";")
df, decay_events = process_co2_decay_events(file_path)
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
