import pandas as pd

# Load the file
file_path = "../coddora_exporter/65m3_export.csv"
df = pd.read_csv(file_path, delimiter=',')

# Parse datetime and sort
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

# Identify presence transitions from True to False
df['presence_shift'] = df['presence'].shift(1)
df['event_start'] = (df['presence_shift'] == True) & (df['presence'] == False)
event_starts = df.index[df['event_start']].tolist()

# Initialize override columns
forty_minutes = pd.Timedelta(minutes=40)
presence_override = pd.Series([False] * len(df))
df['#occupants'] = pd.to_numeric(df['#occupants'], errors='coerce')
df['occupants_modified'] = df['#occupants']

# Apply modifications
for idx in event_starts:
    event_time = df.loc[idx, 'Datetime']
    mask = (df['Datetime'] >= event_time) & (df['Datetime'] < event_time + forty_minutes)
    presence_override[mask] = True

    # Get last known #occupants where presence was True before the event
    prev_true_mask = (df['Datetime'] < event_time) & (df['presence'] == True)
    if not df[prev_true_mask].empty:
        last_occupants = df.loc[prev_true_mask, '#occupants'].iloc[-1]
        df.loc[mask, 'occupants_modified'] = last_occupants

# Final presence modification
df['presence_modified'] = df['presence'] | presence_override

# Optional: save to file
df.to_csv("65m3_export_false_negative.csv", index=False)

# Show result sample
print(df[['Datetime', 'presence', 'presence_modified', '#occupants', 'occupants_modified']].head(10))