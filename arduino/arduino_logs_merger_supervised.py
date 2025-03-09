import os
import pandas as pd
from datetime import datetime, timedelta

file_path = "/media/DATA/uni/thesis/co2data/Salotto/14_02_07_44_15_02_05_59/1402074415020559_supervised_fixed_first.csv"
output_file = "/media/DATA/uni/thesis/co2data/Salotto/14_02_07_44_15_02_05_59/1402074415020559_supervised_fixed_first_fixed.csv"
df = pd.read_csv(file_path, delimiter=',')

start_time = datetime.today().replace(hour=7, minute=44, second=40, microsecond=85049)

df.iloc[:, 2] = df.iloc[:, 3].apply(lambda x: start_time + timedelta(milliseconds=x))
#df.iloc[:, 2] = df.iloc[:, 2].apply(lambda x: start_time + timedelta(milliseconds=int(x.split(".")[1])))
df.to_csv(output_file, index=False,sep=';')

print(f"Sorted and merged log file saved as {output_file}")