import pandas as pd
import h5py
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


input_file_path = "dataset_office_rooms.h5"

def inspect_hdf5(file_path):
    with h5py.File(file_path, 'r') as h5f:
        print("HDF5 File Structure and Details:")
        
        def print_structure(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"Group: {name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"  Dataset: {name}")
                print(f"    Shape: {obj.shape}")
                print(f"    Data Type: {obj.dtype}")
                print(f"    Attributes: {dict(obj.attrs)}")
        
        # Recursively visit all groups and datasets
        h5f.visititems(print_structure)

# Inspect the file
inspect_hdf5(input_file_path)

chunksize = 1000000  # Adjust chunk size based on available memory

# Process metadata
max_sim_id_metadata = 0
for chunk in pd.read_hdf(input_file_path, key='metadata', chunksize=chunksize):
    chunk_max = chunk['simID'].max()
    if chunk_max > max_sim_id_metadata:
        max_sim_id_metadata = chunk_max
print(max_sim_id_metadata)

total_rows = pd.read_hdf(input_file_path, key='metadata', mode='r').shape[0]

    # Read only the last row
last_simId = pd.read_hdf(input_file_path, key='metadata', start=total_rows - 1, stop=total_rows)['simID'].iloc[0]
print(f"First simId: {last_simId}")


last_concentration = pd.read_hdf(input_file_path, key='data', start=total_rows - 1, stop=total_rows)['Zone Air CO2 Concentration'].iloc[0]
print(f"last_concentration: {last_concentration}")

last_occupancy = pd.read_hdf(input_file_path, key='data', start=total_rows - 1, stop=total_rows)['Occupancy'].iloc[0]
print(f"last_occupancy: {last_occupancy}")