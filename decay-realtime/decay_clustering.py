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
    # Compute CO₂ trend
    event_summary["co2_trend"] = decay_events.groupby("event_id")["co2"].diff().mean()

    # Apply conditions: CO2 decay and duration > 30 min
    filtered_events = event_summary[
        (event_summary["start_co2"] > event_summary["end_co2"]) & (event_summary["duration_minutes"] > 30)
    ]
    # Define a threshold to exclude fake decay events
    threshold_factor = 1.4  # Allows a 30% increase from min_co2 but rejects anything higher
    
    # Apply updated decay event conditions (Monotonicity in Decay)
    filtered_events = event_summary[
        (event_summary["start_co2"] > event_summary["end_co2"]) &  # Ensuring initial CO2 is higher than final
        (event_summary["min_co2"] < event_summary["start_co2"]) &  # Ensuring real decay occurred
        (event_summary["end_co2"] < threshold_factor * event_summary["min_co2"]) &  # Avoid cases where end CO2 rises too much
        (event_summary["co2_trend"] < 0)
    ]
    

    # Create a new column 'presence_analysis'
    df["presence_analysis"] = df["presence"].astype(object)
    #df.loc[(df["presence"] == False) & (~df["event_id"].isin(filtered_events.index)), "presence_analysis"] = True
    df.loc[(df["presence"] == False) & (~df["event_id"].isin(filtered_events.index)), "presence_analysis"] = "Ignore"
    
    
    return df, filtered_events

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

def calculate_decay_constants(df, filtered_events):
    df = df.merge(filtered_events[["start_time", "end_time", "start_co2", "end_co2", "cluster"]],
                  left_on="event_id", right_index=True, how="left")

    df["measurement_decay_constant"] = -np.log(df["co2"] / df["start_co2"]) / ((df["date"] - df["start_time"]).dt.total_seconds() / 60)
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
