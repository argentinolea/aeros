import pandas as pd
from datetime import datetime

def fix_datetime_format(dt_str):
    if '.' not in dt_str:
        dt_str += ".000000"

    return dt_str
    
    
# Load the file (update the file path accordingly)
file_path = "/media/DATA/uni/thesis/co2data/Salotto/capture/10022002.csv"
df = pd.read_csv(file_path, delimiter=';')

# Convert the third column (datetime) to pandas datetime format
df.iloc[:, 2] = df.iloc[:, 2].apply(fix_datetime_format)
df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2], format="%Y-%m-%d %H:%M:%S.%f")

# Convert the third column to epoch time in milliseconds
df.iloc[:, 3] = df.iloc[:, 2].apply(lambda x: int(x.timestamp() * 1000))
df.iloc[:, 2] = pd.to_datetime(df.iloc[:, 2]).dt.strftime("%Y-%m-%d %H:%M:%S.%f")


df.to_csv("10022002_time_fixed.csv", index=False, sep=';')