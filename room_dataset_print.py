import pandas as pd
import h5py
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F  # Correct import for PySpark functions


input_file_path = "dataset_office_rooms.h5"

with h5py.File(input_file_path, "r") as hdf:
    data_table = hdf['data/table']
    metadata_table = hdf['metadata/table']
    
    print("Schema for data/table:")
    for name, dtype in data_table.dtype.fields.items():
        print(f"Field: {name}, Data Type: {dtype[0]}")

    print("\nSchema for metadata/table:")
    for name, dtype in metadata_table.dtype.fields.items():
        print(f"Field: {name}, Data Type: {dtype[0]}")
    
with h5py.File(input_file_path, "r") as hdf:
    # Access the `data/table` dataset
    dataset = hdf['data/table']
    
    # Print the first 10 rows
    print("First 10 rows of data/table:")
    for row in dataset[:10]:  # Read the first 10 rows
        print(row)
        
with h5py.File(input_file_path, "r") as hdf:
    # Access the `metadata/table` dataset
    dataset = hdf['metadata/table']
    
    # Print the first 10 rows
    print("First 10 rows of metadata/table:")
    for row in dataset[:10]:  # Read the first 10 rows
        print(row)
               
spark = SparkSession.builder \
    .appName("HDF5 to Spark") \
    .getOrCreate()


output_file_path = "output.h5"

chunk_size = 10000

# Flatten a single chunk of data/table
def flatten_data_table(data_chunk):
    # Extract fields
    index = [row[0] for row in data_chunk]
    values_block_0 = np.array([row[1] for row in data_chunk])
    values_block_1 = np.array([row[2] for row in data_chunk])
    values_block_2 = np.array([row[3][0].decode() for row in data_chunk])  # Decode bytes to strings
    
    # Create DataFrame
    data_df = pd.DataFrame(values_block_0, columns=[f"data_v0_{i}" for i in range(values_block_0.shape[1])])
    data_df[[f"data_v1_{i}" for i in range(values_block_1.shape[1])]] = pd.DataFrame(values_block_1.tolist(), index=data_df.index)
    data_df['data_v2_0'] = values_block_2
    data_df['index'] = index
    return data_df

# Flatten a single chunk of metadata/table
def flatten_metadata_table(metadata_chunk):
    # Handle empty chunks
    if len(metadata_chunk) == 0:
        print("Warning: Empty metadata chunk encountered.")
        return pd.DataFrame(columns=[f"meta_v0_{i}" for i in range(11)] + ["meta_v1_0", "meta_v1_1", "index"])
    
    # Extract fields
    index = [row[0] for row in metadata_chunk]
    values_block_0 = [row[1] for row in metadata_chunk]
    values_block_1 = [row[2] for row in metadata_chunk]
    
    # Dynamically flatten values_block_0
    try:
        values_block_0_df = pd.DataFrame(values_block_0, columns=[f"meta_v0_{i}" for i in range(len(values_block_0[0]))])
    except IndexError:
        print("Warning: Unexpected structure in values_block_0. Skipping chunk.")
        return pd.DataFrame(columns=[f"meta_v0_{i}" for i in range(11)] + ["meta_v1_0", "meta_v1_1", "index"])
    
    # Add values_block_1 and index
    values_block_1_df = pd.DataFrame(values_block_1, columns=["meta_v1_0", "meta_v1_1"])
    values_block_0_df = pd.concat([values_block_0_df, values_block_1_df], axis=1)
    values_block_0_df['index'] = index
    
    return values_block_0_df

# Process HDF5 file in chunks
with h5py.File(input_file_path, 'r') as h5f:
    data_table = h5f['data/table']
    metadata_table = h5f['metadata/table']
    
    for i in range(0, data_table.shape[0], chunk_size):
        # Process chunks of data/table
        data_chunk = data_table[i:i + chunk_size]
        data_df = flatten_data_table(data_chunk)
        data_spark = spark.createDataFrame(data_df)

        # Process corresponding chunks of metadata/table
        metadata_chunk = metadata_table[i:i + chunk_size]
        meta_df = flatten_metadata_table(metadata_chunk)
        if not meta_df.empty:
            meta_spark = spark.createDataFrame(meta_df)
            # Continue processing (e.g., joining with data_spark)
        else:
            print(f"Skipping metadata chunk {i} due to empty DataFrame.")

        # Join on index and meta_v1_1
        merged_spark = data_spark.join(meta_spark, data_spark['index'] == meta_spark['meta_v1_1'], "inner")
        
        # Convert to Pandas for writing to HDF5
        merged_df = merged_spark.toPandas()

        # Append to the output HDF5 file
    with h5py.File(output_file_path, 'a') as h5f_out:
        if "merged/table" not in h5f_out:
            # Define the dataset with appropriate data types
            dtype = [(col, 'S256') if merged_df[col].dtype == 'object' else (col, merged_df[col].dtype) for col in merged_df.columns]
            data_array = np.array([tuple(row) for row in merged_df.itertuples(index=False)], dtype=dtype)
            h5f_out.create_dataset("merged/table", data=data_array, maxshape=(None,), chunks=True)
        else:
            # Resize and append
            current_size = h5f_out["merged/table"].shape[0]
            new_size = current_size + len(merged_df)
            h5f_out["merged/table"].resize(new_size, axis=0)
            h5f_out["merged/table"][current_size:] = np.array([tuple(row) for row in merged_df.itertuples(index=False)], dtype=dtype)