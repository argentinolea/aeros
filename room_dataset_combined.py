from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import h5py
import pandas as pd
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("HDF5Processing").getOrCreate()

# Flatten data/table
def flatten_data_table(data_chunk):
    index = [row[0] for row in data_chunk]
    values_block_0 = np.array([row[1] for row in data_chunk])
    values_block_1 = np.array([row[2] for row in data_chunk])
    values_block_2 = [row[3][0].decode() for row in data_chunk]

    data_df = pd.DataFrame(values_block_0, columns=[f"data_v0_{i}" for i in range(values_block_0.shape[1])])
    data_df[[f"data_v1_{i}" for i in range(values_block_1.shape[1])]] = pd.DataFrame(values_block_1.tolist(), index=data_df.index)
    data_df['data_v2_0'] = values_block_2
    data_df['index'] = index
    return data_df

# Flatten metadata/table
def flatten_metadata_table(metadata_chunk):
    index = [row[0] for row in metadata_chunk]
    values_block_0 = np.array([row[1] for row in metadata_chunk])
    values_block_1 = np.array([row[2] for row in metadata_chunk])

    meta_df = pd.DataFrame(values_block_0, columns=[f"meta_v0_{i}" for i in range(values_block_0.shape[1])])
    meta_df[[f"meta_v1_{i}" for i in range(values_block_1.shape[1])]] = pd.DataFrame(values_block_1.tolist(), index=meta_df.index)
    meta_df['index'] = index
    return meta_df

def prepare_for_hdf5(df):
    for col in df.columns:
        if df[col].dtype == 'O':  # Object type
            # Check for strings
            if df[col].apply(lambda x: isinstance(x, str)).all():
                df[col] = df[col].astype(str).str.encode('utf-8')  # Encode as byte strings
            else:
                raise ValueError(f"Column {col} contains unsupported data types. Review the data.")
    return df

def df_to_hdf5_compatible_array(df):
    dtype = []
    for col in df.columns:
        if df[col].dtype == 'O':
            # Fixed-length string, adjust length as needed
            dtype.append((col, 'S256'))
        else:
            dtype.append((col, df[col].dtype))
    structured_array = np.array([tuple(row) for row in df.to_records(index=False)], dtype=dtype)
    return structured_array

def flatten_metadata_table(metadata_chunk):
    # Check if the chunk is empty
    if len(metadata_chunk) == 0:
        print("Warning: Empty metadata chunk encountered.")
        return pd.DataFrame()

    # Extract fields with error handling
    try:
        index = [row[0] for row in metadata_chunk]
        values_block_0 = [row[1] for row in metadata_chunk]
        values_block_1 = [row[2] for row in metadata_chunk]

        # Ensure values_block_0 is consistent and has elements
        if len(values_block_0) == 0 or not all(isinstance(v, (list, np.ndarray)) for v in values_block_0):
            raise ValueError("Unexpected structure in values_block_0")

        # Flatten values_block_0
        meta_df = pd.DataFrame(values_block_0, columns=[f"meta_v0_{i}" for i in range(len(values_block_0[0]))])
        meta_df[[f"meta_v1_{i}" for i in range(len(values_block_1[0]))]] = pd.DataFrame(values_block_1, index=meta_df.index)
        meta_df['index'] = index
        return meta_df
    except Exception as e:
        print(f"Error processing metadata chunk: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of an error
    
# Process HDF5 file in chunks with Spark
def process_and_save_hdf5(input_file_path, output_file_path, chunk_size=10000):
    with h5py.File(input_file_path, 'r') as h5f:
        data_table = h5f['data/table']
        metadata_table = h5f['metadata/table']

        with h5py.File(output_file_path, 'w') as h5f_out:
            for i in range(0, data_table.shape[0], chunk_size):
                # Flatten data chunk
                data_chunk = data_table[i:i + chunk_size]
                data_df = flatten_data_table(data_chunk)

                # Flatten metadata chunk
                metadata_chunk = metadata_table[i:i + chunk_size]
                meta_df = flatten_metadata_table(metadata_chunk)

                # Skip processing if meta_df is empty
                if meta_df.empty:
                    print(f"Skipping metadata chunk {i} due to empty DataFrame.")
                    continue

                # Continue processing as before
                data_spark = spark.createDataFrame(data_df)
                meta_spark = spark.createDataFrame(meta_df)

                # Concatenate rows
                combined_spark = data_spark.join(meta_spark, on="index")

                # Filter rows where the first column does not match the last column
                filtered_spark = combined_spark.filter(col("data_v0_0") == col("meta_v1_1"))

                # Convert back to Pandas for saving to HDF5
                filtered_df = filtered_spark.toPandas()
                filtered_df = prepare_for_hdf5(filtered_df)
                hdf5_array = df_to_hdf5_compatible_array(filtered_df)

                # Save to output HDF5 incrementally
                if "combined/table" not in h5f_out:
                    h5f_out.create_dataset(
                        "combined/table", 
                        data=hdf5_array, 
                        maxshape=(None,), 
                        chunks=True
                    )
                else:
                    current_size = h5f_out["combined/table"].shape[0]
                    new_size = current_size + len(hdf5_array)
                    h5f_out["combined/table"].resize(new_size, axis=0)
                    h5f_out["combined/table"][current_size:] = hdf5_array

# Input and output paths
input_file_path = "dataset_office_rooms.h5"
output_file_path = "output_combined_file.h5"

# Run the Spark processing
process_and_save_hdf5(input_file_path, output_file_path)