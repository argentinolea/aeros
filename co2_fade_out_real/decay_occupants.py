import pandas as pd
import numpy as np
from datetime import timedelta

# Load the dataset (replace with your actual file path)
file_path = "../capture/10022002_fixed_all_features.csv"
df = pd.read_csv(file_path, delimiter=';')

# Convert column types
df['date'] = pd.to_datetime(df['date'])
df['presence'] = df['presence'].astype(bool)
df['#occupants'] = pd.to_numeric(df['#occupants'], errors='coerce')

# Initialize the exponential decay column
df['exp_decay_#occupants'] = df['#occupants']

# Ventilation rate and decay constant
ventilation_rate = 0.75  # air changes per hour
decay_constant = ventilation_rate / 3600  # per second

# Loop to detect presence transitions and apply decay
i = 1
while i < len(df):
    if df.loc[i - 1, 'presence'] == True and df.loc[i, 'presence'] == False:
        decay_start_idx = i
        decay_start_time = df.loc[i, 'date']
        N0 = df.loc[i - 1, '#occupants']

        j = i
        while j < len(df) and df.loc[j, 'presence'] == False:
            j += 1

        decay_end_idx = j - 1 if j < len(df) else len(df) - 1

        for k in range(decay_start_idx, decay_end_idx + 1):
            elapsed_sec = (df.loc[k, 'date'] - decay_start_time).total_seconds()
            Nt = N0 * np.exp(-decay_constant * elapsed_sec)
            Nt = round(Nt, 2)
            df.loc[k, 'exp_decay_#occupants'] = 0 if Nt < 0.01 else Nt

        i = j
    else:
        i += 1

# (Optional) Save the output
df.to_csv("output_with_truncated_exponential_decay.csv", index=False)

# Print preview
print(df[['date', 'presence', '#occupants', 'exp_decay_#occupants']].head(20))