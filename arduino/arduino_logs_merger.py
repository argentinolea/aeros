import os
import pandas as pd
from datetime import datetime, timedelta

# Define input and output paths
input_folder = "../thesis/co2data/Salotto/10_02_23.15_11_02_20.57/arduino"  # Change to your log file directory
output_file = "../thesis/co2data/Salotto/10_02_23.15_11_02_20.57/2102095921020709.log"

# Define the start time (today at 10:00 AM)
start_time = datetime.today().replace(hour=9, minute=52, second=17, microsecond=0)

# Initialize an empty list to store data
data = []

for filename in os.listdir(input_folder):
    if filename.endswith(".log"):
        filepath = os.path.join(input_folder, filename)
        with open(filepath, "r") as file:
            for line in file:
                parts = line.strip().split("\t")  # Tab-separated values
                if len(parts) == 4:  # Ensure correct format
                    millis = int(parts[0].split(":")[1].strip())  # Extract millis
                    co2 = float(parts[1].split(":")[1].strip())  # Extract CO2
                    temp = float(parts[2].split(":")[1].strip())  # Extract Temperature
                    humidity = float(parts[3].split(":")[1].strip())  # Extract Humidity

                    # Append raw data (millis first for sorting)
                    data.append([millis, co2, temp, humidity])

# Convert to DataFrame
df = pd.DataFrame(data, columns=["Millis", "CO2", "Temp", "Humidity"])

# Sort data by "Millis"
df = df.sort_values(by="Millis").reset_index(drop=True)

# Compute timestamps
df["Timestamp"] = df["Millis"].apply(lambda x: start_time + timedelta(milliseconds=x))

# Reorder columns
df = df[["Timestamp", "Millis", "CO2", "Temp", "Humidity"]]

# Save to CSV
df.to_csv(output_file, index=False)

print(f"Sorted and merged log file saved as {output_file}")