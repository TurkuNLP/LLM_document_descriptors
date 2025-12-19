#!bin/bash

# Concatenate harmonization results from different shards into a single file.

benchmark="$1"
file_dir="../results/harmonized/${benchmark}"
output_dir=${file_dir}/concatenated

mkdir -p "$output_dir"

file_pattern="*_harmonized.jsonl"
concat_out=${output_dir}/"descriptors_${benchmark}_harmonized.jsonl"

echo "Concatenating files into $concat_out"
# Recursively look for files matching the pattern and concatenate them
find "$file_dir" -type f -name "$file_pattern" -exec cat {} + > "$concat_out"

echo "Concatenation complete."