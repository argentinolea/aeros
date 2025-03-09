import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

def shannon_entropy(values):
    """Calculate Shannon entropy from a list of values."""
    counter = Counter(values)
    total = len(values)
    probabilities = np.array([count / total for count in counter.values()])
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def process_co2_entropy(csv_file):
    """Reads a CSV file, groups CO₂ values by presence_analysis, and calculates entropy."""
    df = pd.read_csv(csv_file, delimiter=';')
    
    # Ensure presence_analysis column is treated as boolean
    df['date'] = pd.to_datetime(df['date'])
   # df['presence_analysis'] = df['presence_analysis'].astype(str)
    
    # Group by presence_analysis
    entropy_results = {}
    for presence_analysis_value in ["True", "False"]:
        group_data = df[df['presence_analysis'] == presence_analysis_value]['co2'].tolist()
        if group_data:
            entropy_results[presence_analysis_value] = shannon_entropy(group_data)
    
    return entropy_results


def process_co2_entropy_by_day(csv_file,day):
    """Reads a CSV file, filters data for 01/02/2025, and calculates entropy over time."""
    df = pd.read_csv(csv_file, delimiter=';')
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['presence_analysis'] = df['presence_analysis']
    
    # Filter data only for 01/02/2025
    target_date = day
    df = df[df['date'].dt.date == pd.to_datetime(target_date).date()]
    
    # Sort by time to ensure continuity
    df.sort_values(by='date', inplace=True)
    
    # Compute rolling entropy with a window of 5 minutes
    window_size = '5min'
    entropy_results = {'True': [], 'False': []}
    timestamps = []
    
    for time, group in df.resample(window_size, on='date'):
        timestamps.append(time)
        for presence_analysis_value in ["True", "False"]:
            co2_values = group[group['presence_analysis'] == presence_analysis_value]['co2'].tolist()
            entropy = shannon_entropy(co2_values) if co2_values else np.nan
            entropy_results[str(presence_analysis_value)].append(entropy)
    
    return timestamps, entropy_results

def plot_entropy(timestamps, entropy_results):
    """Plot continuous entropy trends over time for presence_analysis and absence on 01/02/2025."""
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, entropy_results['True'], label='presence_analysis=True', linestyle='-', marker='', linewidth=2)
    plt.plot(timestamps, entropy_results['False'], label='presence_analysis=False', linestyle='-', marker='', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Shannon Entropy')
    plt.title('Shannon Entropy of CO₂ Levels on 01/02/2025')
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    pairplot_output_path = "entropy.png"
    plt.savefig(pairplot_output_path)
    
    
    
if __name__ == "__main__":
    csv_file = "CO2_decay_with_constants.csv"  # Change to your actual file path
    entropy_values = process_co2_entropy(csv_file)
    
    print("Shannon Entropy of CO₂ levels:")
    print(f"presence_analysis=True: {entropy_values.get('True', 'No data')}")
    print(f"presence_analysis=False: {entropy_values.get('False', 'No data')}")
    timestamps, entropy_values_plot = process_co2_entropy_by_day(csv_file,"2025-02-01")
    plot_entropy(timestamps, entropy_values_plot)
    
