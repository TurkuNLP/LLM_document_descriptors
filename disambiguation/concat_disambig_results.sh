#!bin/bash

# Concatenate disambiguation results from different shards into a single file.

file_dir="$1" # Directory containing disambiguation result files
output_dir=${file_dir}/results


disambig_pattern="disambig_*_disambig.jsonl"
disambig_out=${output_dir}/"all_disambig.jsonl"

lineage_pattern="full_lineage.jsonl"
lineage_out=${output_dir}/"all_full_lineage.jsonl"

mkdir -p "$output_dir"

echo "Concatenating disambig files into $disambig_out"
# Recursively look for files matching the pattern and concatenate them
find "$file_dir" -type f -name "$disambig_pattern" -exec cat {} + > "$disambig_out"

echo "Concatenating lineage files into $lineage_out"
find "$file_dir" -type f -name "$lineage_pattern" -exec cat {} + > "$lineage_out"

echo "Concatenation complete."