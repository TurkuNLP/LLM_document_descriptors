import pyarrow.parquet as pq
import argparse
import os

def process_file(input_path, output_path):
    try:
        parquet_file = pq.ParquetFile(input_path)
        mode = 'w'
        
        for batch in parquet_file.iter_batches(batch_size=10000):
            df_chunk = batch.to_pandas()
            df_chunk.to_json(
                output_path, 
                orient='records', 
                lines=True, 
                mode=mode,
                force_ascii=False
            )
            mode = 'a'
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
    print(f"{input_path} done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    args = parser.parse_args()
    process_file(args.input, args.output)